import torch
import matplotlib.pyplot as plt
import numpy as np

def make_prediction_figure(idxs, df, emg, pred_emg):
    for idx in idxs:
        print(idx)
        print(df['meta'][idx])
        
        fig = plt.figure(figsize=(12, 4))
        labs = ['Flexor', 'Extensor']
        for muscle in range(2):
            ax = fig.add_subplot(1, 2, muscle+1)

            ymin, ymax = min(pred_emg[idx, muscle, :].min(), emg[idx, muscle, :].min()), max(pred_emg[idx, muscle, :].max(), emg[idx, muscle, :].max())

            ax.plot(pred_emg[idx, muscle, :], c=plt.get_cmap('tab10')(muscle), linewidth=2)
            ax.plot(emg[idx, muscle, :], '--', c=plt.get_cmap('tab10')(muscle), linewidth=2)
            ax.plot([100, 100],[ymin, ymax], 'k:', alpha=0.5, linewidth=2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([0, 400])
            ax.set_xticklabels(['-100ms', '300ms'], fontweight='bold')

            ax.set_yticks([ymin, ymax])
            ax.set_yticklabels(['%0.2f mV'%(ymin), '%0.2f mV'%(ymax)], fontweight='bold')
            ax.set_title(labs[muscle], fontweight='bold')

            if muscle == 0:
                ax.set_ylabel('EMG', fontsize=12, fontweight='bold')

        plt.savefig('outputs/20200306/beta/many-to-many/deepSpine_100/999/idx{}.png'.format(idx), bbox_inches='tight')
        plt.show()

def draw_matrix(ax, data, xlab, ylab, vmin, vmax):
    im = ax.imshow(data, cmap=plt.get_cmap('Oranges'), vmin=vmin, vmax=vmax)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    y_labs = ['{} Hz'.format(x) for x in xlab]
    x_labs = ['{} uA'.format(x) for x in ylab]

    ax.set_xticks(np.arange(len(xlab)))
    ax.set_xticklabels(x_labs, rotation=45)

    ax.set_yticks(np.arange(len(xlab)))
    ax.set_yticklabels(y_labs)

    return im

# LOAD IN THE META DATA FROM TESTING
df = torch.load('outputs/20200306/beta/many-to-many/deepSpine_100/999/session_0.pth')
pred_emg = df['predicted_emg'].cpu().numpy()
emg = df['emg'].cpu().numpy()


# THIS IS THE MLP TEST DATA
df2 = torch.load('outputs/MLPeval.pth')

uniq_freqs = np.unique(df['meta'][:, 0, 1])
uniq_amps = np.unique(df['meta'][:, 0, 2])

error_mat_deepspine = np.zeros((uniq_freqs.shape[0], uniq_amps.shape[0]), dtype=np.float32)
error_mat_mlp = np.zeros((uniq_freqs.shape[0], uniq_amps.shape[0]), dtype=np.float32)

MLPErr = df2['loss'].mean(dim=(-1,-2))

for fid, freq in enumerate(uniq_freqs):
    for aid, amp in enumerate(uniq_amps):
        idx = np.where(np.logical_and(df2['meta'][:, 0, 1] == freq, df2['meta'][:, 0, 2] == amp))[0]
        
        # i've verified that the validation set is the same in both cases
        ds_loss = df['loss'][idx].cpu().numpy().mean()
        mlp_loss = MLPErr[idx].cpu().numpy().mean()

        error_mat_deepspine[fid, aid] = ds_loss
        error_mat_mlp[fid, aid] = mlp_loss

vmin, vmax = min(error_mat_deepspine.min(), error_mat_mlp.min()), max(error_mat_deepspine.max(), error_mat_mlp.max())

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
_ = draw_matrix(ax, error_mat_deepspine, uniq_freqs, uniq_amps, vmin, vmax)

ax = fig.add_subplot(122)
im = draw_matrix(ax, error_mat_mlp, uniq_freqs, uniq_amps, vmin, vmax)

#cb = fig.colorbar(im, cmap=plt.get_cmap('Oranges'))
#cb.outline.set_visible(False)
#cb.set_ticks([vmin, vmax])

plt.show()

# 50687 is a good one
#171, 230, 254, 
# 678, 1016

idxs = [230, 254, 678, 1016]
##idxs = np.where(df['meta'][:,0,2] == 300.)[0]
#make_prediction_figure(idxs, df, emg, pred_emg)



