import os
import torch
import argparse


def crop_volume(volume, mask):
    """ extract mask area from volume

    :param volume:
    :param mask:
    :return:
    """
    mask_copy = mask.deep_copy()
    imdilate(mask_copy, iteration=5, label=0)

    minbox, maxbox = md.image3d_tools.bounding_box_voxel(mask_copy, min_val=1, max_val=3)
    print(minbox, maxbox)
    minbox[0], minbox[1] = 0, 0
    maxbox[0], maxbox[1] = volume.size()[0], volume.size()[1]

    cropped_image = md.image3d_tools.crop(volume, minbox, maxbox)
    cropped_info = [minbox, maxbox]

    return cropped_image, cropped_info


def inverse_crop(original_img, cropped_img, crop_info, type='image'):
    """ put crop volume back into the original space

    :param volume:
    :param crop_info:
    :param ref:
    :param padding:
    :return:
    """
    [minbox, maxbox] = crop_info
    if type == "image":
        orig_img_arr = original_img.to_numpy()
        orig_img_arr[minbox[2]:maxbox[2], minbox[1]:maxbox[1], minbox[0]:maxbox[0]] = cropped_img.to_numpy()
        original_img.from_numpy(orig_img_arr)
        return original_img
    else:
        # creat zeros and insert back
        original_img = md.image3d_tools.resample_volume_as_ref(cropped_img, original_img)
        orig_img_arr = np.zeros_like(original_img.to_numpy())
        orig_img_arr[minbox[2]:maxbox[2], minbox[1]:maxbox[1], minbox[0]:maxbox[0]] = cropped_img.to_numpy()
        original_img.from_numpy(orig_img_arr)

        return original_img


class AspectsData:
    def __init__(self, im_in, im_seg):
        self.im_aligined_stripped = im_in
        self.im_seg = im_seg
        self.core_index = (0, 0)
        self.output20_shift = {1: 0, 11: 0, 2: 0, 12: 0, 3: 0, 13: 0, 4: 0, 14: 0, 5: 0, 15: 0,
                               6: 0, 16: 0, 7: 0, 17: 0, 8: 0, 18: 0, 9: 0, 19: 0, 10: 0, 20: 0}
        self.output_score_summary = {'left': 10, 'right': 10, 'score': 10}
        self.isCalc = 0

        self.score_mask = None


    def assignImSeg(self, im_seg):
        """
        :param im_seg:
        :return:
        """
        assert isinstance(im_seg, md.Image3d), print("cannot assign value, check image type")
        self.im_seg = im_seg

    def assignScore(self, out_score_20region, output_summary):
        self.output20_shift = out_score_20region
        if output_summary is not None:
            self.output_score_summary = output_summary

    def assignCoreIndex(self, core_ind):
        assert isinstance(core_ind, tuple)
        self.core_index = core_ind

    def assignScoreMask(self, scoremask):
        self.score_mask = scoremask

    def getAspectsReturn(self):
        unwrap = (self.im_aligned,
                  self.im_seg,
                  self.core_index,
                  self.output20_shift,
                  self.output_score_summary,
                  self.isCalc)

        return unwrap

    def get(self, output_dir=None):
        print("#====================#")
        print('out_summary: ', self.output_score_summary)
        print('out_20_score: ', self.output20_shift)
        print('Core Index: ', self.core_index)


class CtBrainAspects(AspectsBase):
    def __init__(self, aspectsdata, model):
        self.aspectsdata = aspectsdata
        self.model = model

        self._crop_info = None

        self.out_dir = self.model.out_dir

    def run(self):
        self._process()
        self._dump()
        return self.aspectsdata

    def _crop(self):
        if self._checkErrorCode():
            im_stripped = self.aspectsdata.im_aligned_stripped
            mask = self.aspectsdata.im_seg
            im_stripped_cropped, crop_info = crop_volume(im_stripped, mask)
            self._crop_info = crop_info

    def _inverseCrop(self):
        if self._checkErrorCode():
            im_aligned_stripped = self.aspectsdata.im_aligned_stripped
            im_seg = self.aspectsdata.im_seg
            im_score_mask = self.aspectsdata.score_mask
            crop_info = self._crop_info

            im_seg = inverse_crop(im_aligned_stripped, im_seg, crop_info, type='mask')
            im_score_mask = inverse_crop(im_aligned_stripped, im_score_mask, crop_info, type='mask')
            self.aspectsdata.assignImSeg(im_seg)
            self.aspectsdata.assignScoreMask(im_score_mask)

    def _post_process(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        if self._checkErrorCode():
            aspects_data = self.aspectsdata

            upper_ind, lower_ind = calculate_core_index(aspects_data.im_seg)
            core_index = (lower_ind, upper_ind)

            seg_shift, output20_shift = shift_label_for_render(im_seg, aspects_data.output20_shift)

            self.aspectsdata.assignCoreIndex(core_index)
            self.aspectsdata.assignScore(output20_shift, output_summary=None)

            # assign isCalc
            self.aspectsdata.isCalc = 1

    def _dump(self):
        if self.out_dir is not None and self._checkErrorCode():
            sitk.WriteImage(self.aspectsdata.im_seg, os.path.join(self.out_dir, 'im_seg.nii.gz'))
            sitk.WriteImage(self.aspectsdata.score_mask, os.path.join(self.out_dir, 'prd_mask.nii.gz'))



def ct_brain_stroke_aspects(input_path, input_model_dir='./models', gpu_id=0, output_dir=None):
    im = sitk.ReadImage(input_path)
    if output_dir is not None and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    total_test_time = 0
    model = torch.load(model_dir=input_model_dir, gpu=gpu_id)

    if input_path.endswith('txt'):
        file_list, seg_list, case_list = read_test_txt(input_path)
    elif input_path.endswith('csv'):
        file_list, seg_list, case_list = read_test_csv(input_path)
    else:
        raise ValueError('image test_list must either be a txt file or a csv file')

    success_cases = 0
    for i, file in enumerate(file_list):
        print('{}: {}'.format(i, file))
        begin = time.time()
        images = [], im_segs = []
        for image_path in file:
            image = sitk.ReadImage(image_path, dtype=np.float32)
            images.append(image)
            seg_path = image_path.replace('.nii,gz', '_seg.nii.gz')
            im_seg = sitk.ReadImage(seg_path, dtype=np.float32)
            im_segs.append(im_seg)
        read_time = time.time() - begin

        aspectsData = AspectsData(im_in=images, im_seg=im_segs)

        begin = time.time()

        ctAspects = CtBrainAspects(aspectsData, model)
        aspects_data = ctAspects.run()
        aspectsReturn = aspects_data.getAspectsReturn()

        test_time = time.time() - begin

        total_time = read_time + test_time
        total_test_time = test_time + total_test_time
        success_cases += 1
        print('read: {:.2f} s, test: {:.2f} s, total: {:.2f} s, avg test time: {:.2f}'.format(
            read_time, test_time, total_time, total_test_time / float(success_cases)))
    
    ctAspects = CtBrainAspects(aspectsData, passPar)
    aspects_data = ctAspects.run()
    # aspects_data.get(output_dir)  # debug line, save HU
    aspectsReturn = aspects_data.getAspectsReturn()
    # write results into output_dir
    To_save(aspectsReturn, output_dir)

    torch.cuda.empty_cache()


def main():
    description = 'Usage: DA-Net \n' \
                  '-i [file] dcm folder or nifti mhd -o [output_directory] \n' \
                  '-m [Aspects_model] \n' \
                  '-g [gpu id]'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', type=str, default=None)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-g', '--gpuid', type=str, default="0")
    parser.add_argument('-m', '--modelpath', type=str,default=None)


    ct_brain_stroke_aspects(args.input, input_model_dir=args.modelpath, gpu_id=int(args.gpuid),
                            output_dir=args.output)

if __name__ == "__main__":
    main()