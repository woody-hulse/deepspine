import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as scistats
from statistics import mean 
import matplotlib.pyplot as plt
import os

class Trainer: 
    def __init__(self, 
            model,
            criterion,
            optimizer,
            data_loader,
            config
        ):
        self.embeddings = model['embeddings']
        self.core = model['core']
        self.readouts = model['readouts']

        print("# of core parameters:", sum(p.numel() for p in self.core.parameters() if p.requires_grad))

        # import pdb
        # pdb.set_trace()
        self.optim_embeddings = optimizer['embeddings']
        self.optim_core = optimizer['core']
        self.optim_readouts = optimizer['readouts']

        self.criterion = criterion

        self.train_loaders = data_loader['train']
        self.valid_loaders = data_loader['valid']

        # init batch iterators
        batch_iterators = []
        for train_loader in self.train_loaders:
            batch_iterator = iter(train_loader)
            batch_iterators.append(batch_iterator)
        self.batch_iterators = batch_iterators

        self.config = config

        self.device = config['device']['type']
        self.n_sessions = len(self.train_loaders)

        trainer_cfg = config['trainer']['args']
        self.max_iter = trainer_cfg['max_iter']
        self.max_norm = trainer_cfg['max_norm']

        self.save_period = trainer_cfg['save_period']
        self.save_dir = trainer_cfg['save_dir']
        self.eval_period = trainer_cfg['eval_period']

        
        if 'is_sliding_window' in trainer_cfg.keys():
            self.is_sliding_window = trainer_cfg['is_sliding_window']
        else:
            self.is_sliding_window = False
        if self.is_sliding_window:
            self.window_size = trainer_cfg['sliding_window_size']
            self.stride = trainer_cfg['stride']

        if 'resume' in trainer_cfg.keys():
            print('resume from %d' % trainer_cfg['resume'])
            state = torch.load(os.path.join(self.save_dir, 'checkpoint_iter{}.pth'.format(trainer_cfg['resume'])))
            self._load_model(state['state_dict'])
            self._load_optim(state['optimizer'])

            self.start_iter = trainer_cfg['resume'] + 1
        else:
            self.start_iter = 0
    
    def _load_model(self, state):
        for embedding, state_dict in zip(self.embeddings, state['embeddings']):
            embedding.load_state_dict(state_dict)

        for readout, state_dict in zip(self.readouts, state['readouts']):
            readout.load_state_dict(state_dict)

        self.core.load_state_dict(state['core'])

    def _load_optim(self, state):
        for optim_embedding, state_dict in zip(self.optim_embeddings, state['embeddings']):
            optim_embedding.load_state_dict(state_dict)

        for optim_readout, state_dict in zip(self.optim_readouts, state['readouts']):
            optim_readout.load_state_dict(state_dict)

        self.optim_core.load_state_dict(state['core'])
            
    def _train_session(self, session, x):
        embedding = self.embeddings[session]
        readout = self.readouts[session]
        optim_embedding = self.optim_embeddings[session]
        optim_readout = self.optim_readouts[session]

        kinematics, emg, ees, meta = x
        
        # [batch x dim x time]
        kinematics = kinematics.to(self.device)
        emg = emg.to(self.device)
        ees = ees.to(self.device)

        embedding.train()
        readout.train()
        
        optim_embedding.zero_grad()
        optim_readout.zero_grad()
        
        # forward
        afferents = embedding(kinematics, ees)        # [batch x dim x time]
        # embedded = embedded.permute([2,0,1])    # [time x batch x dim]
        motor_neurons = self.core(afferents)  # [batch x dim x time]
        predicted_emg = readout(motor_neurons)  # [batch x dim x time]

        # if self.is_sliding_window:
        #     predicted_emg = predicted_emg[:,:,-1:]
            
        loss = self.criterion(predicted_emg, emg)
        loss.backward()

        emb_norm = nn.utils.clip_grad_norm_(embedding.parameters(), self.max_norm)
        readout_norm = nn.utils.clip_grad_norm_(readout.parameters(), self.max_norm)

        optim_embedding.step()
        optim_readout.step()

        return loss.item()

    def _train_iteration(self, iteration):
        self.core.train()
        
        list_session = np.arange(self.n_sessions)
        np.random.shuffle(list_session)
        list_loss = []
        for session in list_session:
            self.optim_core.zero_grad()

            try:
                x = next(self.batch_iterators[session])
            except Exception:
                self.batch_iterators[session] = iter(self.train_loaders[session])
                x = next(self.batch_iterators[session])
            
            loss = self._train_session(session, x)
            list_loss.append(loss)
            
            core_norm = nn.utils.clip_grad_norm_(self.core.parameters(), self.max_norm)
            self.optim_core.step()

        return list_loss

    def _eval_pearsonr(self, output, target, mask=None):
        output_dim = output.size(1)
        T = output.size(2)

        output = output.detach().cpu()                      # [batch x dim x time]
        target = target.detach().cpu()                      # [batch x dim x time]
        
        # poisson_output = torch.distributions.poisson.Poisson(output.exp())
        # sampled_output = poisson_output.sample()
        
        output = output.view(-1, T)                         # [(batch x dim) x time]
        # output = sampled_output.view(-1, T)                         # [(batch x dim) x time]
        target = target.view(-1, T)                         # [(batch x dim) x time]

        if mask is not None:
            mask = mask.detach().cpu()                          # [batch x 1 x time]
            mask = mask.repeat(1, output_dim, 1).view(-1, T)    # [(batch x dim) x time]

            length = mask.sum(1, keepdims=True)                 # [(batch x dim) x 1]
        else:
            mask = 1
            length = T

        masked_output = mask * output
        masked_target = mask * target
        output_mean = masked_output.sum(1, keepdims=True) / length # [(batch x dim) x 1]
        target_mean = masked_target.sum(1, keepdims=True) / length # [(batch x dim) x 1]

        centered_output = mask * (output - output_mean)
        centered_target = mask * (target - target_mean)
        output_std = torch.sqrt(torch.sum(centered_output ** 2, dim=1))
        target_std = torch.sqrt(torch.sum(centered_target ** 2, dim=1))

        cov = torch.sum(centered_output * centered_target, dim=1)
        r = cov / (output_std * target_std)                         # [(batch x dim)]
        r = r.mean().item()

        return r

    def _valid_session(self, iteration, session, is_save=False):  
        embedding = self.embeddings[session]
        readout = self.readouts[session]
        valid_loader = self.valid_loaders[session]

        list_loss = []
        if is_save:
            list_emg, list_predicted_emg, list_meta, list_loss = [], [], [], []

        batch_id = 0
        for kinematics, emg, ees, meta in valid_loader:
            print('BATCH: {}'.format(batch_id))
            batch_id += 1

            # [batch x dim x time]
            kinematics = kinematics.to(self.device)
            emg = emg.to(self.device)
            ees = ees.to(self.device)

            if self.is_sliding_window:
                batch_size, n_channels = kinematics.size(0), kinematics.size(1)
                kinematics = F.unfold(kinematics[:,:,:,None], kernel_size=(self.window_size,1), stride=(self.stride,1))    # Bx(CxW)xN
                kinematics = kinematics.view(batch_size, n_channels, self.window_size, -1)        # BxCxWxN
                n_windows = kinematics.size(3)
                kinematics = kinematics.permute([0,3,1,2])                                    # BxNxCxW
                kinematics = kinematics.contiguous().view(-1, n_channels, self.window_size)       # (BxW)xCxW

                batch_size, n_channels = ees.size(0), ees.size(1)
                ees = F.unfold(ees[:,:,:,None], kernel_size=(self.window_size,1), stride=(self.stride,1))    # Bx(CxW)xN
                ees = ees.view(batch_size, n_channels, self.window_size, -1)        # BxCxWxN
                ees = ees.permute([0,3,1,2])                                    # BxNxCxW
                ees = ees.contiguous().view(-1, n_channels, self.window_size)       # (BxW)xCxW

            embedding.eval()
            readout.eval()
            
            # forward
            afferents = embedding(kinematics, ees)        # [batch x dim x time]
            # embedded = embedded.permute([2,0,1])    # [time x batch x dim]
            motor_neurons = self.core(afferents)  # [batch x dim x time]
            predicted_emg = readout(motor_neurons)  # [batch x dim x time]

            if self.is_sliding_window:
                predicted_emg = predicted_emg.view(batch_size, n_windows, emg.size(1), -1)
                predicted_emg = predicted_emg.permute([0,2,1,3]).contiguous().view(*emg.size())

            loss = self.criterion(predicted_emg, emg, reduction='none')
            loss = loss.mean(dim=(-1,-2))
            #list_loss.append(loss.item())

            if is_save:
                list_emg.append(emg)
                list_predicted_emg.append(predicted_emg)
                list_loss.append(loss)
                list_meta.append(meta)
       
        loss = torch.cat(list_loss, dim=0)
        if is_save:
            emg = torch.cat(list_emg, dim=0)                        # [batch x dim x time]
            predicted_emg = torch.cat(list_predicted_emg, dim=0)    # [batch x dim x time]
            metas = torch.cat(list_meta, dim=0)

            # eval pearson r score
            # pearsonr = self._eval_pearsonr(predicted_emg, emg)
            state = {
                'predicted_emg': predicted_emg,
                'emg': emg,
                'loss': loss,
                'meta': metas
            }

            save_dir = os.path.join(self.save_dir, str(iteration))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)    
        
            filename = os.path.join(save_dir, 'session_%d.pth' % (session))
            torch.save(state, filename)

        import ipdb; ipdb.set_trace()
        return loss#, pearsonr

    def _valid_iteration(self, iteration, is_save=False):
        self.core.eval()
        list_loss, list_pearsonr = [], []

        # fig, ax = plt.subplots(3, self.n_sessions, constrained_layout=False)
        # fig.tight_layout(pad=0) 

        for session in range(self.n_sessions):
            loss = self._valid_session(iteration, session, is_save)
            list_loss.append(loss)
            # list_pearsonr.append(pearsonr)

        # plt.subplots_adjust(wspace=0, hspace=0)
        # fig.savefig('raw_emg_%d.png' % (iteration), dpi=600.)
        # plt.close()

        return list_loss#, list_pearsonr
            
    def _save_checkpoint(self, iteration, save_best=False):
        state = {
            'iteration': iteration,
            'state_dict': {
                "embeddings": [embedding.state_dict() for embedding in self.embeddings],
                "core": self.core.state_dict(),
                "readouts": [readout.state_dict() for readout in self.readouts]
            },
            'optimizer': {
                "embeddings": [optim_embedding.state_dict() for optim_embedding in self.optim_embeddings],
                "core": self.optim_core.state_dict(),
                "readouts": [optim_readout.state_dict() for optim_readout in self.optim_readouts]
            },
            # 'monitor_best': self.mnt_best,
            'config': self.config
        }

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)    
        
        filename = os.path.join(self.save_dir, 'checkpoint_iter{}.pth'.format(iteration))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

        # if save_best:
        #     best_path = str(self.checkpoint_dir / 'model_best.pth')
        #     torch.save(state, best_path)
        #     self.logger.info("Saving current best: model_best.pth ...")


    def train(self):
        train_loss = [[] for _ in range(self.n_sessions)]
        for iteration in range(self.start_iter, self.max_iter):
            
            # import time
            # start_time = time.time()
            iter_loss = self._train_iteration(iteration)
            # print(time.time()-start_time)
            print('iter {}/{}, loss={}'.format(iteration, self.max_iter, iter_loss))

            for session in range(self.n_sessions):
                train_loss[session].append(iter_loss[session])

            if (iteration+1) % self.save_period == 0:
                self._save_checkpoint(iteration)

            if (iteration+1) % self.eval_period == 0:
                
                with torch.no_grad():
                    valid_loss = self._valid_iteration(iteration, is_save=False)

                    log= ''
                    for session, (t_loss, v_loss) in enumerate(zip(train_loss, valid_loss)):
                        
                        log += '[sess: %d][iter: %d]: %02.3f | %02.3f \n' % (session, iteration, mean(t_loss), v_loss)
                

                    log_file = open(os.path.join(self.save_dir, "log.txt"), "a")
                    log_file.write(log) 
                    log_file.close() 

                    # print(valid_loss)

                train_loss = [[] for _ in range(self.n_sessions)]


    def test(self, iteration):
        state = torch.load(os.path.join(self.save_dir, 'checkpoint_iter{}.pth'.format(iteration)))
        self._load_model(state['state_dict'])

        with torch.no_grad():
            valid_loss = self._valid_iteration(iteration, is_save=True)

            log= ''
            
            for session, v_loss in enumerate(valid_loss):
                
                log += '[sess: %d][iter: %d]: %02.3f \n' % (session, iteration, v_loss)
        
            log_file = open(os.path.join(self.save_dir, 'test_log_%d.txt' % (iteration)), "a")
            log_file.write(log) 
            log_file.close() 
