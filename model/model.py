import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import core.metrics as Metrics
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
        self.mlmc_batch_size=64
        self.N0=100
        self.eval_dir=opt['path']['experiments_root']
        if opt['payoff']=='mean':
            print("mean payoff selected.")
            self.payoff = lambda samples: samples #default to identity payoff
        elif opt['payoff']=='second_moment':
            print("second_moment payoff selected.")
            self.payoff = lambda samples: samples**2 #variance/second moment payoff
        else:
            print("opt['payoff'] not recognised. Defaulting to mean calculation.")
            self.payoff = lambda samples: samples
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
        self.image_size=self.opt['model']['diffusion']['image_size']
        self.channels=self.opt['model']['diffusion']['channels']

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
    
    def mc(self, Nl, continous=False):
        self.netG.eval()
        eval_dir=self.eval_dir
        this_sample_dir = os.path.join(eval_dir,'MCsamples')
        if not os.path.exists(this_sample_dir):
            os.mkdir(this_sample_dir)
        l=9
        M=self.M
        with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
            f.write(f'MC params:L={l}, Nsamples={Nl}, M={M}.')
        num_sampling_rounds = Nl // self.mlmc_batch_size + 1
        numrem=Nl % self.mlmc_batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else self.mlmc_batch_size
            if isinstance(self.netG, nn.DataParallel):
                Xf= self.netG.module.mcsample(
                    self.data['SR'], bs, continous)
            else:
                Xf = self.netG.mcsample(
                    self.data['SR'], bs, continous)
            #acts=actspayoff(Xf)
            # Directory to save samples.
            with open(os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                np.savez_compressed(fout, samples=Xf.cpu().numpy())
        
        self.netG.train()
        return None

        
    def Giles_plot(self,acc):
        self.netG.eval()
        #Set mlmc params
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        eval_dir = self.eval_dir
        Nsamples=1000
        condition_x=self.data['SR'].to(self.device)
        min_l=self.min_l
        
        #Variance and mean samples
        sums,sqsums=self.mlmclooper(condition_x,l=1,Nl=1,min_l=0) #dummy run to get sum shapes 
        sums=torch.zeros((Lmax+1-min_l,*sums.shape))
        sqsums=torch.zeros((Lmax+1-min_l,*sqsums.shape))
        # Directory to save means and norms                                                                                               
        this_sample_dir = os.path.join(eval_dir, f"VarMean_M_{M}_Nsamples_{Nsamples}")
        if not os.path.exists(this_sample_dir):
            os.mkdir(this_sample_dir)
            print(f'Proceeding to calculate variance and means with {Nsamples} estimator samples')
            for i,l in enumerate(range(min_l,Lmax+1)):
                print(f'l={l}')
                sums[i],sqsums[i] = self.mlmclooper(condition_x,Nsamples,l)
            
            s=sqsums[:,0].shape
            sumdims=tuple(range(1,len(s))) #sqsums is output of payoff element-wise squared, so reduce     
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
            sums,sqsums,N=self.mlmc(e,condition_x,alpha_0=alpha,beta_0=beta) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
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
            meanimg=Metrics.tensor2img(meanimg,min_max=(0, 1))
            # Write samples to disk or Google Cloud Storage
            with open(os.path.join(this_sample_dir, "meanpayoff.npz"), "wb") as fout:
                np.savez_compressed(fout, meanpayoff=meanimg)
        self.netG.train()

        return None

    def mlmc(self,accuracy,x_in,alpha_0=-1,beta_0=-1):
        accsplit=np.sqrt(.5)
        #Orders of convergence
        alpha=max(0,alpha_0)
        beta=max(0,beta_0)
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        min_l=self.min_l
        L=min_l+2

        mylen=L+1-min_l
        V=torch.zeros(mylen) #Initialise variance vector of each levels' variance
        N=torch.zeros(mylen) #Initialise num. samples vector of each levels' num. samples
        dN=N0*torch.ones(mylen) #Initialise additional samples for this iteration vector for each level
        sqrt_cost=torch.sqrt(M**torch.arange(min_l,L+1.)+torch.hstack((torch.tensor([0.]),M**torch.arange(min_l,1.*L))))
        it0_ind=False
        while (torch.sum(dN)>0): #Loop until no additional samples asked for
            mylen=L+1-min_l
            for i,l in enumerate(torch.arange(min_l,L+1)):
                num=dN[i]
                if num>0: #If asked for additional samples...
                    tempsums,tempsqsums=self.mlmclooper(condition_x=x_in,Nl=int(num),l=l,min_l=min_l) #Call function which gives sums
                    if not it0_ind:
                        sums=torch.zeros((mylen,*tempsums.shape)) #Initialise sums array of unnormed [dX,Xf,Xc], each column is a level
                        sqsums=torch.zeros((mylen,*tempsqsums.shape)) #Initialise sqsums array of normed [dX^2,Xf^2,Xc^2,XcXf], each column is a level
                        it0_ind=True
                    sqsums[i,...]+=tempsqsums
                    sums[i,...]+=tempsums
                    
            N+=dN #Increment samples taken counter for each level
            Yl=imagenorm(sums[:,0])/N
            sumdims=tuple(range(1,len(sqsums[:,0].shape)))
            s=sqsums[:,0].shape
            V=torch.clip(
                (torch.sum(sqsums[:,0],dim=sumdims).squeeze()/np.prod(s[1:]))/N-(Yl)**2
                ,min=0) #Calculate variance based on updated samples
            
            ##Fix to deal with zero variance or mean by linear extrapolation
            #Yl[2:]=torch.maximum(Yl[2:],.5*Yl[1:-1]*M**(-alpha))
            #V[2:]=torch.maximum(V[2:],.5*V[1:-1]*M**(-beta))
            
            #Estimate order of weak convergence using LR
            #Yl=(M^alpha-1)khl^alpha=(M^alpha-1)k(TM^-l)^alpha=((M^alpha-1)kT^alpha)M^(-l*alpha)
            #=>log(Yl)=log(k(M^alpha-1)T^alpha)-alpha*l*log(M)
            X=torch.ones((mylen-1,2))
            X[:,0]=torch.arange(1,mylen)
            a = torch.linalg.lstsq(X,torch.log(Yl[1:]))[0]
            alpha_ = max(-a[0]/np.log(M),0.)
            b = torch.linalg.lstsq(X,torch.log(V[1:]))[0]
            beta_= -b[0]/np.log(M)
            if alpha_0==-1:
                alpha=alpha_
            if beta_0==-1:
                beta=beta_
                
            sqrt_V=torch.sqrt(V)
            Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of samples/level
            dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
            print(f'Asking for {dN} new samples for l=[{min_l,L}]')
            if torch.sum(dN > 0.01*N).item() == 0: #Almost converged
                if max(Yl[-2]/(M**alpha),Yl[-1])>(M**alpha-1)*accuracy*np.sqrt(1-accsplit**2):
                    L+=1
                    print(f'Increased L to {L}')
                    if (L>Lmax):
                        print('Asked for an L greater than maximum allowed Lmax. Ending MLMC algorithm.')
                        break
                    #Add extra entries for the new level and estimate sums with N0 samples 
                    V=torch.cat((V,V[-1]*M**(-beta)*torch.ones(1)), dim=0)
                    sqrt_V=torch.sqrt(V)
                    newcost=torch.sqrt(torch.tensor([M**L+M**((L-1.))]))
                    sqrt_cost=torch.cat((sqrt_cost,newcost),dim=0)
                    Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of sample
                    N=torch.cat((N,torch.tensor([0])),dim=0)
                    dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
                    print(f'With new L, estimate of {dN} new samples for l=[{min_l,L}]')
                    sums=torch.cat((sums,torch.zeros((1,*sums[0].shape))),dim=0)
                    sqsums=torch.cat((sqsums,torch.zeros((1,*sqsums[0].shape))),dim=0)
                    
        print(f'Estimated alpha = {alpha_}')
        print(f'Estimated beta = {beta_}')
        return sums,sqsums,N
    
    def mlmclooper(self,condition_x,Nl,l,min_l=0):
        eval_dir=self.eval_dir
        num_sampling_rounds = Nl // self.mlmc_batch_size + 1
        numrem=Nl % self.mlmc_batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else self.mlmc_batch_size
            if bs==0:
                break
            with torch.no_grad():
                if isinstance(self.netG, nn.DataParallel):
                    Xf,Xc=self.netG.module.mlmcsample(condition_x,bs,l) #should automatically use cuda
                else:
                    Xf,Xc=self.netG.mlmcsample(condition_x,bs,l) #should automatically use cuda
            fine_payoff=self.payoff(Xf)
            coarse_payoff=self.payoff(Xc)
            if r==0:
                sums=torch.zeros((3,*fine_payoff.shape[1:])) #skip batch_size
                sqsums=torch.zeros((4,*fine_payoff.shape[1:]))
            sumXf=torch.sum(fine_payoff,axis=0).to('cpu') #sum over batch size
            sumXf2=torch.sum(fine_payoff**2,axis=0).to('cpu')
            if l==min_l:
                sqsums+=torch.stack([sumXf2,sumXf2,torch.zeros_like(sumXf2),torch.zeros_like(sumXf2)])
                sums+=torch.stack([sumXf,sumXf,torch.zeros_like(sumXf)])
            elif l<min_l:
                raise ValueError("l must be at least min_l")
            else:
                dX_l=fine_payoff-coarse_payoff #Image difference
                sumdX_l=torch.sum(dX_l,axis=0).to('cpu') #sum over batch size
                sumdX_l2=torch.sum(dX_l**2,axis=0).to('cpu')
                sumXc=torch.sum(coarse_payoff,axis=0).to('cpu')
                sumXc2=torch.sum(coarse_payoff**2,axis=0).to('cpu')
                sumXcXf=torch.sum(coarse_payoff*fine_payoff,axis=0).to('cpu')
                sums+=torch.stack([sumdX_l,sumXf,sumXc])
                sqsums+=torch.stack([sumdX_l2,sumXf2,sumXc2,sumXcXf])
    
        # Directory to save samples. Repeatedly overwrites, just to save some example samples for debugging
        if l>min_l:
            this_sample_dir = os.path.join(eval_dir, f"level_{l}")
            if not os.path.exists(this_sample_dir):
                os.mkdir(this_sample_dir)
                samples_f=Metrics.tensor2img(Xf)
                samples_c=Metrics.tensor2img(Xc)            
                with open(os.path.join(this_sample_dir, "samples_f.npz"), "wb") as fout:
                    np.savez_compressed(fout, samplesf=samples_f)
                with open(os.path.join(this_sample_dir, "samples_c.npz"), "wb") as fout:
                    np.savez_compressed(fout, samplesc=samples_c)
                                
        return sums,sqsums 
    
    
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
