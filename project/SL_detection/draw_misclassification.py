# -*- coding: utf-8 -*-
import utils
import numpy as np 
import config
import cv2
import os

def draw(path_img, save_path, preds, gts, nid):
    image = cv2.imread(path_img, 1)

    if preds is None and gts is None:
        pass
    elif preds is None:
        for e in gts:
            e = list(map(int, e))
            cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (0, 0, 255)) 
            cv2.putText(image, "fn", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif gts is None:
        for e in preds:
            cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (255, 0, 0)) 
            cv2.putText(image, "fp", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        for e in gts:
            e = list(map(int, e))
            cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (255, 255, 255)) 
            #cv2.putText(image, "gt", (int(max(2, e[2] - 20)), int(max(2, e[1] + 10))),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 
        overlap = utils.bbox_overlaps(preds, gts)
        idx_assigned_gt = overlap.argmax(axis=1)
        confidence = overlap.max(axis=1)
        assigned_gts = gts[idx_assigned_gt]
        flag = np.where(confidence>=config.iou_thres, 1, 0)
        if np.sum(flag)>0:
            idx_tp = flag.nonzero()[0]
            tps = preds[idx_tp]
            for e in tps:
                cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (0, 255, 0)) 
                cv2.putText(image, "tp", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        flag = np.where(confidence<config.iou_thres, 1, 0)
        if np.sum(flag)>0:
            idx_fp = flag.nonzero()[0]
            fps = preds[idx_fp]
            for e in fps:
                cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (255, 0, 0)) 
                cv2.putText(image, "fp", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                #cv2.putText(image, "{}".format(avg_density[e[1]-1, e[0]-1]), (int(max(2, e[0] - 20)), int(max(2, e[1] - 20))),
                #       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
        tmp = overlap.max(axis=0)
        flag2 = np.where(tmp<config.iou_thres, 1, 0)
        if np.sum(flag2)>0:
            idx_fn = flag2.nonzero()[0]
            fns = gts[idx_fn]
            for e in fns:
                e = list(map(int, e))
                cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (0, 0, 255)) 
                cv2.putText(image, "fn", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #cv2.putText(image, "{}".format(avg_density[e[1]-1, e[0]-1]), (int(max(2, e[0] - 20)), int(max(2, e[1] - 20))),
                #        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 
