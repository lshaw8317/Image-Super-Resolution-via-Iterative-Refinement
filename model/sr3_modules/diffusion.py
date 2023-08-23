import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
import os
from functools import partial
import numpy as np
from tqdm import tqdm

def imagenorm(img):
    s=img.shape
    if len(s)==1: #fix for when img is single dimensional (batch_size,) -> (batch_size,1)
        img=img[:,None]
    n=torch.linalg.norm(torch.flatten(img, start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.sqrt(np.prod(s[1:]))
    return n

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)
        self.M=2
        self.Lmax=11
        self.min_l=3
        self.mlmc_batch_size=128
        self.N0=100
        self.eval_dir='results/sr_sr3_16_128'
        self.payoff = lambda samples: samples #default to identity payoff

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)
    
    @torch.no_grad()
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
        sqrt_cost=torch.sqrt(2*M**(torch.arange(min_l,L+1,dtype=torch.float32)))
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
            a = torch.lstsq(torch.log(Yl[1:]),X)[0]
            alpha_ = max(-a[0]/np.log(M),0.)
            b = torch.lstsq(torch.log(V[1:]),X)[0]
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
                    sqrt_cost=torch.cat((sqrt_cost,torch.tensor([2**(.5)*M**(L/2)])),dim=0)
                    Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of sample
                    N=torch.cat((N,torch.tensor([0])),dim=0)
                    dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
                    print(f'With new L, estimate of {dN} new samples for l=[{min_l,L}]')
                    sums=torch.cat((sums,torch.zeros((1,*sums[0].shape))),dim=0)
                    sqsums=torch.cat((sqsums,torch.zeros((1,*sqsums[0].shape))),dim=0)
                    
        print(f'Estimated alpha = {alpha_}')
        print(f'Estimated beta = {beta_}')
        return sums,sqsums,N

    @torch.no_grad()
    def mlmcsample(self, condition_x, bs, l):
        device=self.betas.device
        x = condition_x[None,...].to(device) #add fake bs
        shape = x.shape
        print(f'condition_x shape={shape}')
        batch_size = bs
        img_f = torch.randn((bs,*shape), device=device)
        print(f'img_f shape={img_f.shape}')
        img_c = img_f.clone().detach().to(device)
        alpha_c=torch.tensor([1.]).to(device)
        dWc=torch.zeros_like(x).to(device)
        numsteps=self.M**l
        maxsteps=self.M**self.Lmax
        stepsize=maxsteps//numsteps
        for t in tqdm(reversed(range(0, numsteps)), desc='sampling loop time step', total=numsteps):
            current_timep1=(t+1)*stepsize
            current_time=t*stepsize
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[current_timep1]]).repeat(batch_size, 1).to(img_f.device)
            
            ftheta = self.denoise_fn(torch.cat([condition_x, img_f], dim=1), noise_level)
            alpha_f=self.alphas_cumprod[current_time]/self.alphas_cumprod_prev[current_time]
            beta_f=1.-alpha_f
            model_mean = torch.sqrt(1./alpha_f)*(img_f-beta_f*ftheta/self.sqrt_one_minus_alphas_cumprod[current_time])
            dWf = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            noise = dWf*torch.sqrt(beta_f)
            img_f = model_mean + noise
            
            alpha_c*=alpha_f
            dWc+=dWf*torch.sqrt(torch.tensor([1./self.M]).to(device))
            if t % self.M == 0:
                ftheta = self.denoise_fn(torch.cat([condition_x, img_c], dim=1), noise_level)
                beta_c=(1.-alpha_c)
                model_mean = torch.sqrt(1./alpha_c)*(img_c-beta_c*ftheta/self.sqrt_one_minus_alphas_cumprod[current_time])
                noise = dWc*torch.sqrt(beta_c)
                img_c = model_mean + noise
            
        return img_f,img_c
    
    @torch.no_grad()
    def mlmclooper(self,condition_x,Nl,l,min_l=0):
        eval_dir=self.eval_dir
        num_sampling_rounds = Nl // self.mlmc_batch_size + 1
        numrem=Nl % self.mlmc_batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else self.mlmc_batch_size
    
            Xf,Xc=self.mlmcsample(condition_x,bs,l) #should automatically use cuda
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
            samples_f=np.clip(Xf.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            samples_f = samples_f.reshape(
                (-1, self.image_size, self.image_size, self.channels))
            samples_c=np.clip(Xc.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            samples_c = samples_c.reshape(
                (-1, self.image_size, self.image_size, self.channels))
            with open(os.path.join(this_sample_dir, "samples_f.npz"), "wb") as fout:
                np.savez_compressed(fout, samplesf=samples_f)
            with open(os.path.join(this_sample_dir, "samples_c.npz"), "wb") as fout:
                np.savez_compressed(fout, samplesc=samples_c)
                                
        return sums,sqsums 
    
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
