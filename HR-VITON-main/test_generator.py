import torch
import torch.nn as nn

from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
import argparse
import os
import time
from cp_dataset_test import CPDatasetTest, CPDataLoader

from networks import ConditionGenerator, load_checkpoint, make_grid
from network_generator import SPADEGenerator
from tensorboardX import SummaryWriter
from utils import *

import torchgeometry as tgm
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm
def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='use amp')
    # Cuda availability
    parser.add_argument('--cuda', default=False, help='Use CUDA (default: False)')

    parser.add_argument('--test_name', type=str, default='test', help='test name')
    parser.add_argument("--dataroot", default="./data/zalando-hd-resize")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--output_dir", type=str, default="./Output")
    parser.add_argument("--datasetting", default="unpaired")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)

    parser.add_argument('--tensorboard_dir', type=str, default='./data/zalando-hd-resize/tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='./eval_models/weights/v0.1/mtviton.pth', help='tocg checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='./eval_models/weights/v0.1/gen.pth', help='G checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    
    # training
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
        
    # Hyper-parameters
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")

    # generator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='most', # normal: 256, more: 512
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

    opt = parser.parse_args()
    return opt

def load_checkpoint_G(model, checkpoint_path, opt):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Force loading to CPU
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    if opt.cuda and torch.cuda.is_available():  # Check CUDA availability
        model.cuda()
    else:
        model.cpu()  # Ensure the model stays on CPU if CUDA is not enabled



def test(opt, test_loader, tocg, generator):
    # Define the device (CPU or GPU)
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA enabled: {opt.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Initialize Gaussian Blur and move to device
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)

    # Move models to the appropriate device
    tocg = tocg.to(device)
    generator = generator.to(device)
    tocg.eval()
    generator.eval()

    # Prepare output directories
    output_dir = opt.output_dir if opt.output_dir else os.path.join('./output', opt.test_name,
                                                                    opt.datamode, opt.datasetting, 'generator',
                                                                    'output')
    grid_dir = os.path.join('./output', opt.test_name, opt.datamode, opt.datasetting, 'generator', 'grid')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    num = 0
    iter_start_time = time.time()

    with torch.no_grad():
        for inputs in test_loader.data_loader:
            # Move inputs to the appropriate device
            pre_clothes_mask = inputs['cloth_mask'][opt.datasetting].to(device)
            parse_agnostic = inputs['parse_agnostic'].to(device)
            agnostic = inputs['agnostic'].to(device)
            clothes = inputs['cloth'][opt.datasetting].to(device)
            densepose = inputs['densepose'].to(device)

            pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.cpu().numpy() > 0.5).astype(float)).to(device)

            # Downsample inputs
            pre_clothes_mask_down = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
            input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
            clothes_down = F.interpolate(clothes, size=(256, 192), mode='bilinear')
            densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')

            # Prepare multi-task inputs
            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

            # Forward pass
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)

            # Warped cloth mask one hot
            warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.cpu().numpy() > 0.5).astype(float)).to(
                device)

            # Process cloth mask composition
            if opt.clothmask_composition != 'no_composition':
                cloth_mask = torch.ones_like(fake_segmap).to(device)
                if opt.clothmask_composition == 'detach':
                    cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                elif opt.clothmask_composition == 'warp_grad':
                    cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                fake_segmap *= cloth_mask

            # Generate fake parse map
            fake_parse_gauss = gauss(
                F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

            # Create one-hot encoding for parse map
            old_parse = torch.zeros(fake_parse.size(0), 13, opt.fine_height, opt.fine_width, device=device)
            old_parse.scatter_(1, fake_parse, 1.0)

            # Labels
            labels = {
                0: ['background', [0]],
                1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
                2: ['upper', [3]],
                3: ['hair', [1]],
                4: ['left_arm', [5]],
                5: ['right_arm', [6]],
                6: ['noise', [12]]
            }
            parse = torch.zeros(fake_parse.size(0), 7, opt.fine_height, opt.fine_width, device=device)
            for i, (_, indices) in labels.items():
                for label in indices:
                    parse[:, i] += old_parse[:, label]

            # Generate warped cloth
            N, _, iH, iW = clothes.shape
            flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
            flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)],
                                  3)
            grid = make_grid(N, iH, iW, opt).to(device)
            warped_grid = grid + flow_norm
            warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
            warped_clothmask = F.grid_sample(pre_clothes_mask, warped_grid, padding_mode='border')

            if opt.occlusion:
                warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                warped_cloth = warped_cloth * warped_clothmask + torch.ones_like(warped_cloth) * (1 - warped_clothmask)

            # Generator output
            output = generator(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse)

            # Save output images
            unpaired_names = []
            for i in range(output.shape[0]):
                unpaired_name = f"output_{num + i:05d}.png"
                unpaired_names.append(unpaired_name)
            save_images(output, unpaired_names, output_dir)

            num += output.shape[0]
            print(f"Processed {num} images")

    print(f"Test time: {time.time() - iter_start_time:.2f} seconds")


def main():
    print("Starting test_generator script...")

    opt = get_opt()
    print("Start to test! - HR-VITON")

    # Check and set device
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA enabled: {opt.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids if opt.cuda else ""

    # Create test dataset and loader
    test_dataset = CPDatasetTest(opt)
    test_loader = CPDataLoader(opt, test_dataset)

    ## Model
    # Initialize tocg
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96,
                              norm_layer=nn.BatchNorm2d)
    tocg = tocg.to(device)  # Move tocg to the appropriate device

    # Initialize generator
    opt.semantic_nc = 7
    generator = SPADEGenerator(opt, 3 + 3 + 3)  # Densepose (3), Parse-agnostic (3), Warped cloth (3)
    generator = generator.to(device)  # Move generator to the appropriate device
    generator.print_network()

    # Load checkpoints
    load_checkpoint(tocg, opt.tocg_checkpoint, opt)
    load_checkpoint_G(generator, opt.gen_checkpoint, opt)

    # Run the test function
    test(opt, test_loader, tocg, generator)

    print("Finished testing!")


if __name__ == "__main__":
    main()