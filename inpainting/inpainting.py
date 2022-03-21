import os, sys
import numpy as np
import cvbase as cvb
import subprocess as sp

from inpainting.tools.frame_inpaint import DeepFillv1

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

def init_args(args):
    args.dataset_root = None

    # FlowNet2
    args.FlowNet2 = True
    args.pretrained_model_flownet2 = "./pretrained_models/FlowNet2_checkpoint.pth.tar"
    args.img_size = [512, 832]
    args.rgb_max = 255.
    args.fp16 = False
    args.data_list = None
    file_name, ext = os.path.splitext(os.path.basename(args.src))
    args.frame_dir = "./result/" + file_name + "_frame"
    args.PRINT_EVERY = 50

    # DFCNet
    args.DFC = True
    args.ResNet101 = True
    args.MS = False
    args.batch_size = 1
    args.n_threads = 16
    args.get_mask = False
    args.output_root = None
    args.DATA_ROOT = None
    args.MASK_ROOT = None
    args.FIX_MASK = False
    args.MASK_MODE = None
    args.SAVE_FLOW = False
    args.IMAGE_SHAPE = [240, 424]
    args.RES_SHAPE = [240, 424]
    args.GT_FLOW_ROOT = None
    args.PRETRAINED_MODEL = None
    args.PRETRAINED_MODEL_1 = "./pretrained_models/resnet101_movie.pth"
    args.PRETRAINED_MODEL_2 = None
    args.PRETRAINED_MODEL_3 = None
    args.INITIAL_HOLE = False
    args.EVAL_LIST = None
    args.enlarge_mask = False
    args.enlarge_kernel = 10

    # Flow-Guided Propagation
    args.Propagation = True
    args.img_shape = [480, 840]
    args.th_warp = 40
    args.img_root = None
    args.mask_root = None
    args.flow_root = None
    args.output_root_propagation = None
    args.pretrained_model_inpaint = "./pretrained_models/imagenet_deepfill.pth"

def extract_flow(args):
    from inpainting.tools.infer_flownet2 import infer
    output_file = infer(args)
    flow_list = [x for x in os.listdir(output_file) if '.flo' in x]
    flow_start_no = min([int(x[:5]) for x in flow_list])

    zero_flow = cvb.read_flow(os.path.join(output_file, flow_list[0]))
    cvb.write_flow(zero_flow*0, os.path.join(output_file, '%05d.rflo' % flow_start_no))
    args.DATA_ROOT = output_file

def flow_completion(args):

    data_list_dir = os.path.join(args.dataset_root, 'data')
    if not os.path.exists(data_list_dir):
        os.makedirs(data_list_dir)
    initial_data_list = os.path.join(data_list_dir, 'initial_test_list.txt')
    print('Generate datalist for initial step')

    from inpainting.dataset.data_list import gen_flow_initial_test_mask_list
    gen_flow_initial_test_mask_list(flow_root = args.DATA_ROOT,
                                    output_txt_path = initial_data_list)
    args.EVAL_LIST = os.path.join(data_list_dir, 'initial_test_list.txt')

    from inpainting.tools.test_scripts import test_initial_stage
    args.output_root = os.path.join(args.dataset_root, 'Flow_res', 'initial_res')
    args.PRETRAINED_MODEL = args.PRETRAINED_MODEL_1

    if args.img_size is not None:
        args.IMAGE_SHAPE = [args.img_size[0] // 2, args.img_size[1] // 2]
        args.RES_SHAPE = args.IMAGE_SHAPE

    print('Flow Completion in First Step')
    test_initial_stage(args)
    args.flow_root = args.output_root

    if args.MS:
        args.ResNet101 = False
        from inpainting.tools.test_scripts import test_refine_stage
        args.PRETRAINED_MODEL = args.PRETRAINED_MODEL_2
        args.IMAGE_SHAPE = [320, 600]
        args.RES_SHAPE = [320, 600]
        args.DATA_ROOT = args.output_root
        args.output_root = os.path.join(args.dataset_root, 'Flow_res', 'stage2_res')

        stage2_data_list = os.path.join(data_list_dir, 'stage2_test_list.txt')
        from inpainting.dataset.data_list import gen_flow_refine_test_mask_list
        gen_flow_refine_test_mask_list(flow_root = args.DATA_ROOT,
                                       output_txt_path = stage2_data_list)
        args.EVAL_LIST = stage2_data_list
        test_refine_stage(args)

        args.PRETRAINED_MODEL = args.PRETRAINED_MODEL_3
        args.IMAGE_SHAPE = [480, 840]
        args.RES_SHAPE = [480, 840]
        args.DATA_ROOT = args.output_root
        args.output_root = os.path.join(args.dataset_root, 'Flow_res', 'stage3_res')

        stage3_data_list = os.path.join(data_list_dir, 'stage3_test_list.txt')
        from inpainting.dataset.data_list import gen_flow_refine_test_mask_list
        gen_flow_refine_test_mask_list(flow_root = args.DATA_ROOT,
                                       output_txt_path = stage3_data_list)
        args.EVAL_LIST = stage3_data_list
        test_refine_stage(args)
        args.flow_root = args.output_root

def flow_guided_propagation(args):

    deepfill_model = DeepFillv1(pretrained_model = args.pretrained_model_inpaint,
                                image_shape = args.img_shape)

    from inpainting.tools.propagation_inpaint import propagation
    propagation(args, frame_inapint_model = deepfill_model)

def createVideoClip(clip, folder, name, size=[256, 256]):
    vf = clip.shape[0]
    command = ["ffmpeg",
                "-y",  # overwrite output file if it exists
                "-f", "rawvideo",
                "-s", "%dx%d" % (size[1], size[0]),  # "256x256", # size of one frame
                "-pix_fmt", "rgb24",
                "-r", "25",  # frames per second
                "-an",  # Tells FFMPEG not to expect any audio
                "-i", "-",  # The input comes from a pipe
                "-vcodec", "libx264",
                "-b:v", "1500k",
                "-vframes", str(vf),  # 5*25
                "-s", "%dx%d" % (size[1], size[0]),  # "256x256", # size of one frame
                folder + "/" + name]
    # sfolder+"/"+name
    pipe = sp.Popen(command, stdin = sp.PIPE, stderr = sp.PIPE)
    out, err = pipe.communicate(clip.tostring())
    pipe.wait()
    pipe.terminate()
    print(err)

def inpaint(args):
    init_args(args)

    if args.frame_dir is not None:
        args.dataset_root = os.path.dirname(args.frame_dir)
    if args.FlowNet2:
        extract_flow(args)

    if args.DFC:
        flow_completion(args)

    # set propagation args
    assert args.mask_root is not None or args.MASK_ROOT is not None
    args.mask_root = args.MASK_ROOT if args.mask_root is None else args.mask_root
    args.img_root = args.frame_dir

    if args.output_root_propagation is None:
        args.output_root_propagation = os.path.join(args.dataset_root, 'Inpaint_Res')
    if args.img_size is not None:
        args.img_shape = args.img_size
    if args.Propagation:
        flow_guided_propagation(args)

    final_clip = np.stack(args.output_frames)
    video_path = args.dataset_root
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    file_name, ext = os.path.splitext(os.path.basename(video_path))

    createVideoClip(final_clip, video_path, "%s.mp4" % (file_name), [args.img_shape[0], args.img_shape[1]])
    print("Predicted video clip saving")
