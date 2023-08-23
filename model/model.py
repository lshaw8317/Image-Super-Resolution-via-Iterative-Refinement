import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import os
from model.sr3_modules.diffusion import imagenorm
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        
        self.M=2
        self.Lmax=11
        self.min_l=3
        self.mlmc_batch_size=128
        self.N0=100
        self.eval_dir='results/sr_sr3_16_128'
        self.payoff = lambda samples: samples #default to identity payoff
        kwargs={'M':self.M,'Lmax':self.Lmax,'min_l':self.min_l,
                'mlmc_batch_size':self.mlmc_batch_size,'N0':self.N0,
                'eval_dir':self.eval_dir,'payoff':self.payoff}
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt,**kwargs))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()
        
    def Giles_plot(self,acc):
        self.netG.eval()
        #Set mlmc params
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        eval_dir = self.eval_dir
        Nsamples=10**3
        condition_x=self.data['SR']
        min_l=self.min_l
        with torch.no_grad():
    
            #Variance and mean samples
            sums,sqsums=self.netG.module.mlmclooper(condition_x,l=1,Nl=1,min_l=0) #dummy run to get sum shapes 
            sums=torch.zeros((Lmax+1-min_l,*sums.shape))
            sqsums=torch.zeros((Lmax+1-min_l,*sqsums.shape))
            # Directory to save means and norms                                                                                               
            this_sample_dir = os.path.join(eval_dir, f"VarMean_M_{M}_Nsamples_{Nsamples}")
            if not os.path.exists(this_sample_dir):
                os.mkdir(this_sample_dir)
                print(f'Proceeding to calculate variance and means with {Nsamples} estimator samples')
                for i,l in enumerate(range(min_l,Lmax+1)):
                    print(f'l={l}')
                    sums[i],sqsums[i] = self.netG.module.mlmclooper(condition_x,Nsamples,l)
    
                sumdims=tuple(range(1,len(sqsums[:,0].shape))) #sqsums is output of payoff element-wise squared, so reduce     
                s=sqsums[:,0].shape
                means_dp=imagenorm(sums[:,0])/Nsamples
                V_dp=(torch.sum(sqsums[:,0],dim=sumdims).squeeze()/np.prod(s[1:]))/Nsamples-means_dp**2  
            
                # Write samples to disk or Google Cloud Storage
                with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                    torch.save(sums/Nsamples,fout)
                with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                    torch.save(sqsums/Nsamples,fout)
                with open(os.path.join(this_sample_dir, "Ls.pt"), "wb") as fout:
                    torch.save(torch.arange(min_l,Lmax+1,dtype=torch.int32),fout)
                
                #Estimate orders of weak (alpha from means) and strong (beta from variance) convergence using LR
                X=np.ones((Lmax-min_l,2))
                X[:,0]=np.arange(min_l+1,Lmax+1)
                a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
                alpha = -a[0]/np.log(M)
                b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
                beta = -b[0]/np.log(M) 
    
                print(f'Estimated alpha={alpha}\n Estimated beta={beta}\n')
                with open(os.path.join(this_sample_dir, "mlmc_info.txt"),'w') as f:
                    f.write(f'MLMC params: N0={N0}, Lmax={Lmax}, Lmin={min_l}, Nsamples={Nsamples}, M={M}.\n')
                    f.write(f'Estimated alpha={alpha}\n Estimated beta={beta}')
                with open(os.path.join(this_sample_dir, "alphabeta.pt"), "wb") as fout:
                    torch.save(torch.tensor([alpha,beta]),fout)
                    
            with open(os.path.join(this_sample_dir, "alphabeta.pt"),'rb') as f:
                temp=torch.load(f)
                alpha=temp[0].item()
                beta=temp[1].item()
            
            #Do the calculations and simulations for num levels and complexity plot
            for i in range(len(acc)):
                e=acc[i]
                print(f'Performing mlmc for accuracy={e}')
                sums,sqsums,N=self.netG.module.mlmc(e,condition_x,alpha_0=alpha,beta_0=beta) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
                sumdims=tuple(range(1,len(sqsums[:,0].shape))) #sqsums is output of payoff element-wise squared, so reduce
                s=sqsums[:,0].shape
    
                # Directory to save means, norms and N
                dividerN=N.clone() #add axes to N to broadcast correctly on division
                for i in range(len(sums.shape[1:])):
                    dividerN.unsqueeze_(-1)
                this_sample_dir = os.path.join(eval_dir, f"M_{M}_accuracy_{e}")
                
                if not os.path.exists(this_sample_dir):
                    os.mkdir(this_sample_dir)        
                with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                    torch.save(sums/dividerN,fout)
                with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                    torch.save(sqsums/dividerN,fout) #sums has shape (L,4,C,H,W) if img (L,4,2048) if activations
                with open(os.path.join(this_sample_dir, "N.pt"), "wb") as fout:
                    torch.save(N,fout)
    
                meanimg=torch.sum(sums[:,0]/dividerN[:,0,...],axis=0)#cut off one dummy axis
                meanimg=np.clip(meanimg.permute(1, 2, 0).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                # Write samples to disk or Google Cloud Storage
                with open(os.path.join(this_sample_dir, "meanpayoff.npz"), "wb") as fout:
                    np.savez_compressed(fout, meanpayoff=meanimg)
        self.netG.train()

        return None

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
