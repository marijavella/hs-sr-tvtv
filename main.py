from TVTVHS_Solver import TVTVHS_Solver
import argparse
import scipy.io as sio
from utils import*


parser = argparse.ArgumentParser()

parser.add_argument('--r_dir', type=str, default='./data/response coefficient CVPR.mat')  # path for R function
parser.add_argument('--w-dir', type=str, default='./data/W/')   # path for the outputs from base-method W (HRHS from base-method)
parser.add_argument('--gt-dir', type=str, default='./data/X/')  # path for the ground-truth images X (HRHS)
parser.add_argument('--y-dir', type=str, default='./data/Y/')   # path for the HRMS images Y (HRMS)
parser.add_argument('--z-dir', type=str, default='./data/Z/')   # path for the LRHS images Z (LRHS)
parser.add_argument('--outputs-dir', type=str, default='Results/')

# settings for TV-TV Solver
parser.add_argument('--scaling-factor', type=int, default=32)  # scaling factor
parser.add_argument('--beta', type=float, default=1.0)  # beta for ADMM 

args = parser.parse_args()

if not os.path.exists(args.w_dir):
    os.makedirs(args.w_dir)

if len(os.listdir(args.w_dir)) == 0:
    get_W()  # download data from Google drive

# Call each of the testing images individually and process each band individually
for root, dirs, files in os.walk('./data/X'):
    files.sort()

    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for i in range(1, len(files)+1):
        print('Image', i,  'processing')

        inZ = sio.loadmat((args.z_dir + files[i-1]))['Zmsi']
        W = sio.loadmat((args.w_dir + files[i-1]))['outX']
        inX = sio.loadmat((args.gt_dir + files[i-1]))['msi']
        inY = sio.loadmat(args.y_dir+ files[i-1])['RGB']
        R = sio.loadmat(args.r_dir)['R']

        nbands = inX.shape[2]  # no. of bands

        M = inX.shape[0]
        N = inX.shape[1]
        m = inZ.shape[0]
        n = inZ.shape[1]

        Z = inZ.reshape((-1, nbands), order='F')
        w = W.reshape((-1, nbands), order='F')
        w = np.clip(w, 0, 1)
        Z = np.expand_dims(Z, 2)
        W = np.expand_dims(w, 2)
        inY = np.dot(inX, R)
        y = inY.reshape((-1, 3), order='F')

        # Call TV-TV Solver
        x_solver_parallel = TVTVHS_Solver(M, N, Z, W, args.beta, args.scaling_factor, R, y)
        x_solver = (x_solver_parallel).reshape((M, N, nbands), order='F')

        x_solver = np.clip(x_solver, 0, 1)
        W = np.squeeze(W).reshape((M, N, nbands), order='F')

        # Compute quality metrics
        evaluate_metrics(inX, x_solver, W, args.outputs_dir, args.scaling_factor)

        band = 15  # band to plot
        plot_results(inX, x_solver, W, band)

    # Print results
    print('********* Mean PSNR, SSIM, SAM, ERGAS and RMSE Results **********')

    print('****************** TV-TV *******************')

    print('psnr_tv = ', np.load('Results/mean_psnr_solver.npy')[0], 'dB')
    print('ssim_tv = ', np.load('Results/mean_ssim_solver.npy')[0])
    print('sam_tv = ', np.load('Results/mean_sam_solver.npy')[0])
    print('rmse_tv = ', np.load('Results/mean_rmse_solver.npy')[0])
    print('ergas_tv = ', np.load('Results/mean_ergas_solver.npy')[0])

    print('****************** Network *******************')

    print('psnr_network = ', np.load('Results/mean_psnr_cnn.npy')[0], 'dB')
    print('ssim_network = ', np.load('Results/mean_ssim_cnn.npy')[0])
    print('sam_network = ', np.load('Results/mean_sam_cnn.npy')[0])
    print('rmse_network = ', np.load('Results/mean_rmse_cnn.npy')[0])
    print('ergas_network = ', np.load('Results/mean_ergas_cnn.npy')[0])
