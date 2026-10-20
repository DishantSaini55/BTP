import argparse
import torch
import torch.utils.data
import os
import numpy as np
try:
    from . import baselineUtils, individual_TF, context_TF
except ImportError:  # Supports the original scripts run from MTN/.
    import baselineUtils
    import individual_TF
    import context_TF
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(model_path=None):
    traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': ['obd_speed','ego_op_flow','ped_op_flow'],
                       'prediction_type': ['bbox']
                       }
    parser = argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Checkpoint to evaluate (required unless passed programmatically).')
    parser.add_argument('--model_opts', type=dict, default=traj_model_opts)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--ff_size', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_length', type=int, default=15)
    parser.add_argument('--context', action='store_true',
                        help='Evaluate the IDD-PeD scene/occlusion context variant.')
    parser.add_argument('--gate_mode', choices=['hard', 'soft'], default='hard')
    parser.add_argument('--ego_patches', type=int, default=64)
    parser.add_argument('--ped_patches', type=int, default=9)
    args = parser.parse_args()
    if not args.model_path:
        parser.error('--model_path is required')
    inp_l = args.input_length
    ego_patches = args.ego_patches
    ped_patches = args.ped_patches
    device = torch.device("cuda")
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    mean_tr, std_tr,mean_speed,std_speed = baselineUtils.generate_mean_std()
    test_dataset=baselineUtils.create_iddp_dataset(mean_tr,std_tr,mean_speed,std_speed,dataset='iddp',flag='test',include_context=args.context,**args.model_opts)

    model_class = context_TF.ContextIndividualTF if args.context else individual_TF.IndividualTF
    model_kwargs = dict(N=args.layers, d_model=args.emb_size, d_ff=args.ff_size,
                        h=args.heads, dropout=args.dropout)
    if args.context:
        model_kwargs['gate_mode'] = args.gate_mode
    model = model_class(inp_l,4, 4, 4, **model_kwargs).to(device)
    total_params=0
    for params in model.parameters():
        num_params=1
        for x in params.size():
            num_params *= x
        total_params += num_params
    print("total parameters: {}".format(total_params))
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    def split_resultbytime(test_result):
        time_loss = []
        for i in range(45):
            print('time:{},CMSE:{}'.format(i+1, test_result[:, i, :].mean(axis=None)))
            time_loss.append(test_result[:, i, :].mean(axis=None))

    def split_result_by_time_ranges(test_result):
        # averages = []
        
        for time_range in [15, 30, 45]:
            avg = test_result[:, :time_range, :].mean(axis=(0, 1, 2))
            # averages.append(avg)
            print(f'Average MSE for first {time_range} time steps: {avg}')
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    with torch.no_grad():
        model.eval()
        mse_center_list = []
        mse_list = []

        for id_b, batch in enumerate(tqdm(test_dl)):
            inp = batch['enc_input'].float().to(device)
            ego_op_flow = batch['ego_op_flow'].permute(0, 1, 3, 2)
            ego_flow = ego_op_flow[:, :(inp_l-1)].reshape(ego_op_flow.shape[0], 2*(inp_l-1), -1).to(device)
            ped_op_flow = batch['ped_op_flow'].permute(0, 1, 3, 2)
            ped_flow = ped_op_flow[:, :(inp_l-1)].reshape(ego_op_flow.shape[0], 2*(inp_l-1), -1).to(device)
            obd_speed = batch['obd_speed'][:, :inp_l].to(device)
            out_input = batch['enc_input'][:, -1, :].unsqueeze(1).repeat(1, 45, 1).float().to(device)
            dec_inp = torch.Tensor([0, 0, 0, 0]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 45, 1).to(device)

            src_mask = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            obd_enc_mask = torch.ones((inp.shape[0], 1, 1)).to(device)
            spd_mask = torch.ones((inp.shape[0], 1, ego_patches)).to(device)
            ped_mask = torch.ones((inp.shape[0], 1, ped_patches)).to(device)

            if args.context:
                out = model(out_input, inp,obd_speed,ego_flow,ped_flow,dec_inp, src_mask,obd_enc_mask,spd_mask,ped_mask,
                            batch['scene_context'].to(device), batch['observed_occlusion'][:, :inp_l].to(device))
            else:
                out = model(out_input, inp,obd_speed,ego_flow,ped_flow,dec_inp, src_mask,obd_enc_mask,spd_mask,ped_mask)

            out = out.to(device).cpu().numpy()
            gth = batch['pred_target'].to(device).cpu().numpy()

            gthtraj = np.add(np.multiply(gth, std_tr), mean_tr)
            predtraj = np.add(np.multiply(out, std_tr), mean_tr)

            gthtraj_center=np.concatenate((np.expand_dims((gthtraj[:,:,0]+gthtraj[:,:,2])/2,axis=-1),np.expand_dims((gthtraj[:,:,1]+gthtraj[:,:,3])/2,axis=-1)),axis=-1)
            predtraj_center = np.concatenate((np.expand_dims((predtraj[:, :, 0] + predtraj[:, :, 2]) / 2, axis=-1),np.expand_dims((predtraj[:, :, 1] + predtraj[:, :, 3]) / 2, axis=-1)), axis=-1)
            performance = np.square(gthtraj - predtraj)
            mse_center_list.append(np.square(gthtraj_center-predtraj_center))
            mse_list.append(performance)

        mse = np.concatenate(mse_list, axis=0)
        mse_center=np.concatenate(mse_center_list,axis=0)
        print('mse_example_loss',mse.shape)
        # split_resultbytime(mse)
        split_result_by_time_ranges(mse)
        split_resultbytime(mse_center)
        print('MSE:',mse.mean(axis=None),'CMSE:',mse_center.mean(axis=None))
        exit()



if __name__=='__main__':
    test()
