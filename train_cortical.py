import time
import shutil
import argparse
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import SimpleITK as sitk



cortical_regions = [5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20]
subcor_regions = [1, 2, 3, 4, 11, 12, 13, 14]
cortical = [str(c) for c in cortical_regions]
subcor = [str(s) for s in subcor_regions]

region_dict = {
    'C_RR': 1,
    'C_LL': 11,
    'I_RR': 2,
    'I_LL': 12,
    'IC_RR': 3,
    'IC_LL': 13,
    'L_RR': 4,
    'L_LL': 14,
    'M1_RR': 5,
    'M1_LL': 15,
    'M2_RR': 6,
    'M2_LL': 16,
    'M3_RR': 7,
    'M3_LL': 17,
    'M4_RR': 8,
    'M4_LL': 18,
    'M5_RR': 9,
    'M5_LL': 19,
    'M6_RR': 10,
    'M6_LL': 20
}


def kaiming_init(net):
    net.apply(kaiming_weight_init)


def save_intermediate_results(idxs, crops, masks, outputs, file_names, region_ids, tags, oups, weights, out_folder):
    """ save intermediate results to training folder

    :param idxs: the indices of crops within batch to save
    :param crops: the batch tensor of image crops
    :param masks: the batch tensor of segmentation crops
    :param outputs: the batch tensor of output label maps
    :param frames: the batch frames
    :param file_names: the batch file names
    :param out_folder: the batch output folder
    :return: None
    """
    #print(type(edge),type(frames),type(outputs))
    #print(edge.shape)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    for i in idxs:

        case_out_folder = os.path.join(out_folder, file_names[i])
        if not os.path.isdir(case_out_folder):
            os.makedirs(case_out_folder)

        region_id = region_ids[i]
        tag_l = tags[0][i].item()
        tag_r = tags[1][i].item()

        oup_l = oups[0][i].item()
        oup_r = oups[1][i].item()

        weight_l = weights[0][i].item()
        weight_r = weights[1][i].item()

        if crops is not None:
            images = ToImage()(crops[0][i])
            for modality_idx, image in enumerate(images):
                sitk.WriteImage(os.path.join(case_out_folder, 'crop_left_%d_r%d_w%.2f_t%d_o%.3f.nii' % (
                                                    modality_idx, region_id, weight_l, tag_l, oup_l)), image)
            images = ToImage()(crops[1][i])
            for modality_idx, image in enumerate(images):
                sitk.WriteImage(image, os.path.join(case_out_folder, 'crop_right_%d_r%d_w%.2f_t%d_o%.3f.nii' % (
                                                modality_idx, region_id, weight_r, tag_r, oup_r)), image)

        if masks is not None:
            mask = ToImage()(masks[0][i, 0].short())
            sitk.WriteImage(os.path.join(case_out_folder, 'mask_left_r%d.nii' % region_id), mask)
            mask = ToImage()(masks[1][i, 0].short())
            sitk.WriteImage(os.path.join(case_out_folder, 'mask_right_r%d.nii' % region_id), mask)

        if outputs is not None:
            output = ToImage()(outputs[0][i, 0].data)
            sitk.WriteImage(os.path.join(case_out_folder, 'att_left_r%d.nii.gz' % region_id), output)
            output = ToImage()(outputs[1][i, 0].data)
            sitk.WriteImage(os.path.join(case_out_folder, 'att_right_r%d.nii.gz' % region_id), output)


def train(config_file):
    assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_module_from_disk(config_file)
    cfg = cfg.cfg

    root_dir = os.path.dirname(config_file)
    cfg.general.im_list = os.path.join(root_dir, cfg.general.im_list)
    cfg.general.save_dir = os.path.join(root_dir, cfg.general.save_dir)

    def adjust_learning_rate(optimizer, epoch):
        lr = cfg.train.lr * (0.7 ** (epoch // 6))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # control randomness during training
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)

    # clean the existing folder if not continue training
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        sys.stdout.write("Found non-empty save dir.\n"
                         "Type 'yes' to delete, 'no' to continue: ")
        choice = input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError("Please type either 'yes' or 'no'!")

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'classification')

    # enable CUDNN
    cudnn.benchmark = True

    dataset = ClassificationDataset(
        imlist_file=cfg.general.im_list,
        tag_list_file=cfg.general.tag_list,
        mask=cfg.dataset.mask,
        num_classes=cfg.dataset.num_classes,
        crop_size=cfg.dataset.crop_size,
        spacing=cfg.dataset.spacing,
        region_id=cfg.general.region,
        default_values=cfg.dataset.default_values,
        random_translation=cfg.dataset.random_translation,
        interpolation=cfg.dataset.interpolation,
        normalizers=cfg.dataset.crop_normalizers,
        cbf_ratio_file=cfg.general.cbf_file,
        subcor=False)

    sampler = EpochConcateSampler(dataset, cfg.train.epochs)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batchsize,
                             num_workers=cfg.train.num_threads, pin_memory=True)

    gpu_ids = list(range(cfg.general.num_gpus))
    net = ClassificationNet(2, use_nwu=cfg.net.nwu, fc_layer=2, k=None)

    net = nn.parallel.DataParallel(net, device_ids=gpu_ids)
    net = net.cuda()

    opt = optim.Adam(net.parameters(), lr=cfg.train.lr, betas=cfg.train.betas)

    if cfg.general.resume_epoch >= 0:
        net = torch.load(os.pth.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.cfg.general.resume_epoch))
        last_save_epoch, batch_start = cfg.general.resume_epoch, 0
    else:
        last_save_epoch, batch_start = 0, 0

    batch_idx = batch_start
    data_iter = iter(data_loader)

    for i in range(len(data_loader)):
        epoch_idx = batch_idx * cfg.train.batchsize // len(dataset)
        adjust_learning_rate(opt, epoch_idx)
        begin_t = time.time()

        images_left, images_right, mask_left, mask_right, mask_depress_left, mask_depress_right, tag_left, \
        tag_right, weight_left, weight_right, hu_left, hu_right, region_id, case_name = data_iter.next()

        images_left, images_right, mask_left, mask_right, mask_depress_left, mask_depress_right = \
            images_left.cuda(), images_right.cuda(), mask_left.cuda(), mask_right.cuda(), \
            mask_depress_left.cuda(), mask_depress_right.cuda()

        inp_left = torch.cat([images_left, mask_left], dim=1)
        inp_right = torch.cat([images_right, mask_right], dim=1)

        tag_bi = torch.clamp(tag_left + tag_right, max=1)
        weight_bi = (weight_left + weight_right) / 2
        #mask_bi = torch.clamp(mask_left + mask_right, max=1)

        tag_left, tag_right, tag_bi = \
            tag_left.cuda().unsqueeze(1), tag_right.cuda().unsqueeze(1), tag_bi.cuda().unsqueeze(1)
        weight_left, weight_right, weight_bi = \
            weight_left.cuda().unsqueeze(1), weight_right.cuda().unsqueeze(1), weight_bi.cuda().unsqueeze(1)
        hu_left, hu_right = hu_left.cuda().unsqueeze(1), hu_right.cuda().unsqueeze(1)
        nwu_right = (1 - hu_right / (hu_left + 1e-4))
        nwu_left = (1 - hu_left / (hu_right + 1e-4))

        if cfg.net.nwu is True:
            out_l, out_r, cam_l, cam_r = net(inp_left, inp_right, nwu_left.float(), nwu_right.float())
        else:
            out_l, out_r, cam_l, cam_r = net(inp_left, inp_right)

        weight_l_tag = cal_true_label_weight(tag_left) * weight_left
        weight_l_tag = weight_l_tag / torch.mean(weight_l_tag)
        weight_r_tag = cal_true_label_weight(tag_right) * weight_right
        weight_r_tag = weight_r_tag / torch.mean(weight_r_tag)

        weight_l_pred = cal_pred_label_weight(out_l) * weight_left
        weight_l_pred = weight_l_pred / torch.mean(weight_l_pred)
        weight_r_pred = cal_pred_label_weight(out_r) * weight_right
        weight_r_pred = weight_r_pred / torch.mean(weight_r_pred)

        loss_l = nn.BCELoss(weight=weight_l_tag.float())
        loss_r = nn.BCELoss(weight=weight_r_tag.float())
        loss_l_pred = nn.BCELoss(weight=weight_l_pred.float())
        loss_r_pred = nn.BCELoss(weight=weight_r_pred.float())

        loss_mse = WeightedMSE()
        mse_l = loss_mse(cam_l, mask_depress_left)
        mse_r = loss_mse(cam_r, mask_depress_right)

        loss_rank = RankLoss()
        rank = loss_rank(out_l, out_r, tag_left, tag_right, weight_left, weight_right)

        loss_avg = AvgLoss()
        avg = loss_avg(out_l, out_r)

        print(tag_left.squeeze(1), '\n', out_l.squeeze(1))
        print(tag_right.squeeze(1), '\n', out_r.squeeze(1))

        bce_l = loss_l(out_l, tag_left.float())
        bce_r = loss_r(out_r, tag_right.float())
        bce_l_pred = loss_l_pred(out_l, tag_left.float())
        bce_r_pred = loss_r_pred(out_r, tag_right.float())

        print('bce: ', bce_l.item(), bce_r.item(), bce_l_pred.item(), bce_r_pred.item())
        print('mse: ', mse_l.item(), mse_r.item())
        print('rank: ', rank.item())
        print('avg: ', avg.item())

        w_bce = (bce_l + bce_r) / (bce_l_pred + bce_r_pred)
        w_bce = w_bce.detach()
        w_pred = 1.4 if torch.max(out_l.detach()) > 0.5 and torch.max(out_r.detach()) > 0.5 else 0

        train_loss = (bce_l * w_bce + bce_r * w_bce + bce_l_pred * w_pred + bce_r_pred * w_pred) / (w_bce + w_pred) \
                     + (mse_l + mse_r) * 1 + rank + 0.5 * avg

        train_loss.backward()
        opt.step()

        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batchsize

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration)
        logger.info(msg)

        if (batch_idx + 1) % cfg.train.plot_snapshot == 0:
            train_loss_plot_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
            plot_loss(log_file, train_loss_plot_file, name='train_loss',
                      display='Training Loss ({})'.format(cfg.loss.name))

        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                torch.save(net, os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx)))
                last_save_epoch = epoch_idx

        if cfg.debug.save_inputs and (batch_idx + 1) % 100 == 0:
            save_intermediate_results(list(range(inp_left.shape[0])), [images_left.cpu(), images_right.cpu()],
                                      [mask_left.cpu(), mask_depress_right.cpu()], [cam_l.cpu(), cam_r.cpu()],
                                      file_names=case_name, region_ids=region_id,
                                      out_folder=cfg.general.save_dir, tags=[tag_left.cpu(), tag_right.cpu()],
                                      oups=[out_l.cpu(), out_r.cpu()], weights=[weight_left.cpu(), weight_right.cpu()])


def main():

    long_description = "DenseNet Classification Train Engine"

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', nargs='?', default="/home/yichu/ASPECTS/config/config.py",
                        help='volumetric segmentation3d train config file')
    args = parser.parse_args()
    train(args.input)


if __name__ == '__main__':
    main()







