import argparse
import time
from pathlib import Path
import sys
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from tools.utils.datasets import LoadStreams, LoadImages
from tools.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from tools.utils.plots import plot_one_box
from tools.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from models.yolo import Model
from models.common import DetectMultiBackend
sys.path.insert(0, './yolov7')
# sys.path.insert(0, './yoloair-iscyy-beta/')
print(sys.path)
sys.path.append('.')

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # model = DetectMultiBackend(weights=opt.weights, device=device, dnn=opt.dnn, data=opt.data)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        visualize = opt.visualize
        mulpicplus = "1"     #1 for normal,2 for 4pic plus,3 for 9pic plus and so on
        assert(int(mulpicplus) >= 1)
        if mulpicplus == "1":
            pred = model(img, augment=opt.augment, visualize=increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False)[0]

        else:
            xsz = img.shape[2]
            ysz = img.shape[3]
            mulpicplus = int(mulpicplus)
            x_smalloccur = int(xsz / mulpicplus * 1.2)
            y_smalloccur = int(ysz / mulpicplus * 1.2)
            for i in range(mulpicplus):
                x_startpoint = int(i * (xsz / mulpicplus))
                for j in range(mulpicplus):
                    y_startpoint = int(j * (ysz / mulpicplus))
                    x_real = min(x_startpoint + x_smalloccur, xsz)
                    y_real = min(y_startpoint + y_smalloccur, ysz)
                    if (x_real - x_startpoint) % 64 != 0:
                        x_real = x_real - (x_real-x_startpoint) % 64
                    if (y_real - y_startpoint) % 64 != 0:
                        y_real = y_real - (y_real - y_startpoint) % 64
                    dicsrc = img[:, :, x_startpoint:x_real,
                                                    y_startpoint:y_real]
                    pred_temp = model(dicsrc, augment=opt.augment, visualize=increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False)[0]
                    pred_temp[..., 0] = pred_temp[..., 0] + y_startpoint
                    pred_temp[..., 1] = pred_temp[..., 1] + x_startpoint
                    if i == 0 and j == 0:
                        pred = pred_temp
                    else:
                        pred = torch.cat([pred, pred_temp], dim=1)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        results = []

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Run tracker
            detections = []
            if len(det):
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes

            online_targets = tracker.update(detections, im0)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    # save results
                    results.append(
                        f"{dataset.count},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

                    # print(results)
                    # res_file = opt.project + '/' + opt.name + ".txt"
                    res_file = str(save_dir) + '/' + opt.name + ".txt"
                    with open(res_file, 'a') as f:
                        f.writelines(results)

                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=1)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Stream results

            if view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    '''
    res_file = opt.project + '/' + opt.name + ".txt"
    with open(res_file, 'w') as f:
        f.writelines(results)

    print(f"save results to {res_file}")
    '''
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='D:/ycr/proj/yoloair-iscyy-beta/runs/train/yolov7-C3HBinC3C5/weights/last.pt', help='model.pt path(s)')
    # D: / ycr / proj / BoT - SORT - main / tools / runs / train / exp32 / weights / last.pt
    parser.add_argument('--source', type=str, default='D:/ycr/proj/BoT-SORT-main/tools/datasets/mot/car/012/img/', help='source')  # file/folder, 0 for webcam

    parser.add_argument('--data', type=str, default='D:/ycr/proj/BoT-SORT-main/yolov7/data/VISO.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.001, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=True, help='display results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', default=False, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='D:/ycr/proj/BoT-SORT-main/tools/datasets/mot/car/012/', help='save results to project/name')
    parser.add_argument('--name', default='yolov7-C3HBinC3C5', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')
    parser.add_argument('--visualize', action='store_true', help='visualize features')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=50, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"D:/ycr/proj/BoT-SORT-main/fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"D:/ycr/proj/BoT-SORT-main/pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
