import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from collections import OrderedDict


def imagenorm(img):
    s=img.shape
    if len(s)==1: #fix for when img is single dimensional (batch_size,) -> (batch_size,1)
        img=img[:,None]
    n=torch.linalg.norm(torch.flatten(img, start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.sqrt(np.prod(s[1:]))
    return n

def Giles_plot(diff,acc):
        #Set mlmc params
        M=diff.M
        N0=diff.N0
        Lmax=diff.Lmax
        Nsamples=10**3
        
        min_l=diff.min_l

        #Variance and mean samples
        sums,sqsums,_=diff.mlmc(1e5,M,N0=1,min_l=0) #dummy run to get sum shapes 
        sums=torch.zeros((Lmax+1-min_l,*sums.shape[1:]))
        sqsums=torch.zeros((Lmax+1-min_l,*sqsums.shape[1:]))
        condition_x=diff.data['SR']
        # Directory to save means and norms                                                                                               
        this_sample_dir = os.path.join(diff.eval_dir, f"VarMean_M_{M}_Nsamples_{Nsamples}")
        if not os.path.exists(this_sample_dir):
            os.mkdir(this_sample_dir)
            print(f'Proceeding to calculate variance and means with {Nsamples} estimator samples')
            for i,l in enumerate(range(min_l,Lmax+1)):
                print(f'l={l}')
                sums[i],sqsums[i] = diff.mlmclooper(condition_x,Nsamples,l)

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
            with open(os.path.join(this_sample_dir, "mlmc_info.txt"),'wb') as f:
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
            sums,sqsums,N=diff.mlmc(e,alpha_0=alpha,beta_0=beta) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
            sumdims=tuple(range(1,len(sqsums[:,0].shape))) #sqsums is output of payoff element-wise squared, so reduce
            s=sqsums[:,0].shape

            # Directory to save means, norms and N
            dividerN=N.clone() #add axes to N to broadcast correctly on division
            for i in range(len(sums.shape[1:])):
                dividerN.unsqueeze_(-1)
            this_sample_dir = os.path.join(diff.eval_dir, f"M_{M}_accuracy_{e}")
            
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
       
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        assert phase == 'val'
        val_set = Data.create_dataset(dataset_opt, phase)
        val_loader = Data.create_dataloader(
            val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    logger.info('Initial Model Finished')
    #Modify noise schedule to correspond to MLMC max L in diffusion
    opt['model']['beta_schedule'][opt['phase']]['n_timestep']=diffusion.M**diffusion.Lmax
    
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    logger.info('Begin Model Evaluation.')
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = diffusion.eval_dir
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        #val_data automatically has batch size 1 for phase=val
        idx += 1
        diffusion.feed_data(val_data) #loads in self.data['SR'] which is accessed by self.mlmc
        acc=[.1,.05,.01,.005]
        Giles_plot(diffusion,acc)
        visuals=OrderedDict()
        visuals['INF'] = diffusion.data['SR'].detach().float().cpu()
        visuals['HR'] = diffusion.data['HR'].detach().float().cpu()
        if 'LR' in diffusion.data:
            visuals['LR'] = diffusion.data['LR'].detach().float().cpu()

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        Metrics.save_img(
            hr_img, '{}/hr.png'.format(result_path))
        Metrics.save_img(
            lr_img, '{}/lr.png'.format(result_path))
        Metrics.save_img(
            fake_img, '{}/inf.png'.format(result_path))

