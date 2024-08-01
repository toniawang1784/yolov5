# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torchsummary import summary
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
import torch
torch.cuda.empty_cache()
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(os.path.join(ROOT,'../AIOB')) not in sys.path:
    sys.path.append(str(os.path.join(ROOT, '../AIOB')))
from AIOB import AIOB
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders_savevid import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, LoadDicom
# from utils.dataloaders_batchIMAGE import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, LoadDicom
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        batchsz=1, # inference batch size
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
    
    # Get GA cases
    # print(source)
    # root = source
    # studies = os.listdir(root)
    # studies = [each for each in studies if len(os.listdir(os.path.join(root,each)))>0]
    # df_trueGA = AIOB.trueGA_df().query('age_true/7>20')
    # exams_trueGA = df_trueGA.index.tolist()
    # ga_studies = []
    # for existing_exam in studies:
    #     if existing_exam.replace("-butterfly","") in exams_trueGA or existing_exam.replace("-novice-butterfly","") in exams_trueGA:
    #         ga_studies.append(existing_exam)
    # print('Original number of files: ', len(studies))
    # print('GA filter number of files: ', len(ga_studies))
    # existing_dir = '/home/pj-024018/rest/YOLO_FAMLI_video_v1'
    # existing_studies = os.listdir(existing_dir)
    # studies_to_inference = [each for each in ga_studies if each not in existing_studies]
    # dirs_to_inference = [os.path.join(root, study) for study in studies_to_inference]
    # print(len(studies_to_inference))
    # cases = os.listdir(source)
    # df_GAtrue = AIOB.trueGA_df().query("age_true/7>=20")
    # exams_GA = df_GAtrue.index.tolist()
    # cases_to_inference = [each for each in cases if each.replace('-butterfly','') in exams_GA or each.replace('-novice-butterfly','') in exams_GA]
    # half_done_case = ['FAM-025-0366-2'] 
    # half_done_case_indicator = half_done_case in cases_to_inference
    # existing_case = os.listdir('/home/pj-024018/rest/YOLO_FAMLI_video_v1')
    # cases_to_inference = [case for case in cases_to_inference if case not in existing_case]
    # if half_done_case_indicator:
    #     cases_to_inference.append(half_done_case)
    # dirs_to_inference = [os.path.join(source, each) for each in cases_to_inference]
    # Directories
    # save_dir = Path('/home/pj-024018/rest/YOLO_FAMLI_video_v1')  # increment run
    name = source.replace('/DICOMS/Sweeps/Primary','').rsplit('/',2)[-1]
    save_dir = increment_path(Path(project)/name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # stride, names, pt = model.stride, model.names, model.pt
    pt = True
    names = {0: 'AmnioticFluid', 1: 'Placenta', 2: 'Head', 3: 'Spine', 4: 'Heart', 5: 'Stomach', 6: 'UrinaryBladder', 7: 'UmbilicalVein'}
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 16
    # studyname = source.rsplit('/', 2)[-1]
    # txt_path = str('/local/data/FAMLI/yolo_inference_csvs/{}.csv'.format(studyname))  # im.txt
    # df_study = pd.read_csv(txt_path).to_numpy()
    # sop = path.rsplit('/')[-1]
    # pred = df_study[df_study[:, 10]==sop]
    # sweep = np.unique(pred[:,1]).tolist()
    # assert len(sweep)==1
    # if sweep not in possibleLateralSweeps_sorted+possibleVerticalSweeps_sorted: continue
    # print(pred)
    # Dataloader
    for study_path in tqdm(studies_to_inference, total=len(studies_to_inference)):
        # source = os.path.join(root, studyname)
        source = study_path
        # save_dir = increment_path(Path(project)/name, exist_ok=exist_ok) 
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadDicom(source,bs=bs,img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride, device=device)
            # dataset = LoadImages(source, batch_size = bs, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # if dataset == False: continue
        # tot_frames = dataset.frames
        vid_path, vid_writer = None, None # 1 writer per frame, 
        # vid_path, vid_writer = [None] * bs, [None] * bs
        current_sweep = ''
        sweep_file_dirs = []
        for path, im, im0s, vid_cap, s in dataset:
            # studyname = source.rsplit('/', 2)[-1]
            # split_path = path.rsplit('/')[-1].split('_')
            # studyname = split_path[0]
            # sweep = split_path[1]
            # txt_frame = split_path[2].replace('.jpg','')
            # frame = int(txt_frame)
            txt_path = str('/local/data/FAMLI/yolo_inference_csvs/{}.csv'.format(studyname))  # im.txt
            df_study = pd.read_csv(txt_path)
            numpy_study = df_study.to_numpy()
            sop = path.rsplit('/')[-1]
            pred = numpy_study[numpy_study[:, 10]==sop]
            sweep = np.unique(pred[:,1]).tolist()
            assert len(sweep)==1
            frame = getattr(dataset, 'frame', 0)
            # print(sweep[0], dataset.frames, frame)
            frames_in_batch=np.linspace(frame-im0s.shape[0], frame, im0s.shape[0], endpoint=False, dtype=int)
            pred_grouped_by_frame = []
            for eachframe in frames_in_batch:
                numpy_frames = pred[pred[:,2]==(eachframe+1)]
                pred_grouped_by_frame.append(numpy_frames.tolist())
            assert len(pred_grouped_by_frame)==len(frames_in_batch)
            # pred = numpy_study[numpy_study[:,1]==sweep]
            # pred = pred[pred[:,2]==(frame+1)]
            if len(pred.shape)==2:
                pred = np.expand_dims(pred,0)
            
            sweep = sweep[0]
            savename = studyname+'_'+sweep
            save_path = str(save_dir / studyname)  # im.jpg
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            img_save_path = os.path.join(save_path, 'imgs'+sweep)
            if not os.path.exists(img_save_path):
                os.mkdir(img_save_path)
            vid_save_path = os.path.join(save_path, 'vids')
            if not os.path.exists(vid_save_path):
                os.mkdir(vid_save_path)

            # Process predictions
            # for i, det in enumerate(pred):
            for i, det in enumerate(pred_grouped_by_frame):  # per image
                # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                # det = np.asarray(det)

                p, im0 = path[i], im0s[i].copy()
                frame=frames_in_batch[i]
                if len(im0.shape)>3:
                    p = Path(p)  # to Path
                    im0 = im0[i]
                else:
                    p = Path(p)
                # s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    det = np.asarray([each[3:9] for each in det]) # has to be array or tensor to plot bbox correctly
                    for cls, conf, *xywh in det:
                        xywh = [float(each) for each in xywh]
                        conf = float(conf)
                        xyxy = (xywh2xyxy(torch.tensor(xywh).view(1, 4)) * gn).view(-1).tolist()  # normalized xywh
                        xyxy = [int(each) for each in xyxy]
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int([key for key in names.keys() if names[key]==cls][0])  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.name}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                save_file_name = savename+'_'+str(frame)
                sweep_file_dir = Path(str(img_save_path)+'/'+save_file_name+'.png').with_suffix('.png')
                cv2.imwrite(sweep_file_dir,im0)
                # sweep_file_dirs.append(im0)
                
                sweepvid_save_path = str(vid_save_path)+'/'+savename+'.mp4'
                if vid_path != sweepvid_save_path:  # new video
                    vid_path = sweepvid_save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                        vid_writer=None
                    fps, w, h = 20, im0.shape[1], im0.shape[0]
                    sweepvid_save_path = str(Path(sweepvid_save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                # else:
                #     fps, w, h = 10, im0.shape[1], im0.shape[0]
                #     vid_save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                #     vid_writer[frame] = cv2.VideoWriter(vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                # print(im0.shape[1], im0.shape[0])
                vid_writer.write(im0)
                # if view_img:
                #     if platform.system() == 'Linux' and p not in windows:
                #         windows.append(p)
                #         cv2.namedWinp.mandow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #     cv2.imshow(str(p), im0)
                #     cv2.waitKey(1)  # 1 millisecond

                # # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video' or 'stream'
                        # if vid_path[i] != save_path:  # new video
                        #     vid_path[i] = save_path
                        #     if isinstance(vid_writer[i], cv2.VideoWriter):
                        #         vid_writer[i].release()  # release previous video writer
                        #     fps, w, h = 10, im0.shape[1], im0.shape[0]
                        #     print(frame)
                        #     save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        #     vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        # vid_writer[i].write(im0)

            # Print time (inference-only)
        #     LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # # Print results
        # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(bs, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--batchsz', '--bs', '--batch-size', type=int, default=1, help='inference batch size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # print(args)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    start = time.time()
    opt = parse_opt()
    main(opt)
    end = time.time()
    #Subtract Start Time from The End Time
    total_time = end - start
    # print("\n"+ str(total_time))