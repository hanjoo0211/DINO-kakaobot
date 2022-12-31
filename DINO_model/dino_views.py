from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse

import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

from PIL import Image
import datasets.transforms as T
import urllib
import base64
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

# config 파일 경로
model_config_path = "../DINO_model/config/DINO/DINO_4scale.py"
# 학습 체크포인트 파일 경로
model_checkpoint_path = "../DINO_model/ckpts/20_checkpoint_best_regular.pth"

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# id 클래스 대치 파일
with open('../DINO_model/util/20class_plant_coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}


def dino_api(request):
    utterance = json.loads(request.body)['userRequest']['utterance']
    # try:
    if request.method == 'POST':
        imageURL = utterance
        openedURL = urllib.request.urlopen(imageURL)
        with open('image.jpeg', 'wb') as f:
            f.write(openedURL.read())

        image = Image.open('image.jpeg').convert("RGB")
        imgDPI = 96
        imgWidth, imgHeight = image.size

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(image, None)

        model.eval()
        with torch.no_grad():
            output = model.cuda()(image[None].cuda())
            output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

        threshold = 0.2 # threshold 조절

        vslzr = COCOVisualizer()

        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > threshold
        print(scores[select_mask])

        # 같은 bbox에 중복 label 제거
        dupBoxes = boxes[select_mask].tolist()
        for i, bbox in enumerate(dupBoxes):
            if bbox in dupBoxes[:i]:
                select_mask[i] = False

        box_label = [id2name[int(item)] for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes[select_mask],
            'size': torch.Tensor([image.shape[1], image.shape[2]]),
            'box_label': box_label
        }
        vslzr.visualize(image, pred_dict, savedir='./', dpi=imgDPI, show_in_console=False, width=imgWidth, height=imgHeight)
        print(box_label)
        print(pred_dict['boxes'])

        # etc 및 중복 버튼 제거
        labelJson = []
        dup = []
        for i in range(len(box_label)):
            if box_label[i] not in dup and box_label[i] != 'etc':
                labelJson.append((box_label[i], str(scores[i])[9:11]))
                dup.append(box_label[i])

        # Result
        result = ''
        for l, s in labelJson:
            result += l + ' ' + s + '%\n'

        # API JSON
        sendJson = {}
        sendJson['version'] = '2.0'

        imageIP = os.environ.get('IMAGE_IP')
        simpleImage = {}
        simpleImage['imageUrl'] = imageIP
        simpleImage['altText'] = '이미지 로딩 실패'
        simpleImageJson = {'simpleImage': simpleImage}

        simpleTextJson = {}
        simpleTextJson['simpleText'] = {'text': result[:-1]}

        sendJson['template'] = {'outputs': [simpleImageJson, simpleTextJson]}

        return JsonResponse(sendJson)
'''
    except:
        if request.method == 'POST':
            # API JSON
            sendJson = {}
            sendJson['version'] = '2.0'

            simpleImageJson = {}
            # simpleImageJson['imageUrl'] = resultImgURL
            simpleImageJson['simpleText'] = {'text': '식물 이미지를 올려주세요!'}

            sendJson['template'] = {'outputs': [simpleImageJson]}
            # sendJson['contents'] = [{'text': '식물 이미지를 올려주세요!', 'type': 'text', 'forwardable': True}]

            return JsonResponse(sendJson)
'''