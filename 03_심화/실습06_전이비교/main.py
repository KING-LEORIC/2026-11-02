from pathlib import Path
import sys
# 현재 실행 중인 파일의 디렉터리 경로
ROOT=Path(__file__).resolve().parent
# 프로젝트 루트 디렉터리(day11_image_classification)
PROJECT=ROOT.parent/'day11_image_classification'
# 사용자 정의 모듈 import를 위해 경로 추가
sys.path.insert(0,str(PROJECT))
import torch
from torch import nn
import pandas as pd
from models import SmallCNN, build_model, set_training_mode
from data import load_data

# 실험 재현성을 위한 시드 고정
torch.manual_seed(42)
# CPU 스레드 수 고정
torch.set_num_threads(1)
# 결과 저장 폴더 생성
out=ROOT/'outputs';out.mkdir(exist_ok=True)

# 전이 모델은 실제 사전학습 가중치를 사용한다. 없으면 다운로드하며 무작위 대체하지 않는다.
# BEGIN D06
# TODO D06: SmallCNN·Head·Partial 세 경로를 실행한다.

# (모델 종류, width, epoch 수)
SETTINGS=[('small',16,1), ('head',16,1), ('partial',16,1)]
# END D06

from models import set_training_mode
from train import evaluate
import copy,time
# 모델별 최종 결과 저장
rows=[]
for kind,width,epochs in SETTINGS:
    # 각 실험이 같은 조건에서 시작되도록 시드 재설정
    torch.manual_seed(42)
    # 모델 종류에 맞는 데이터셋 및 데이터로더 생성
    # - small: 32×32 이미지 사용
    # - head/partial: ResNet 입력 크기(224×224) 사용
    train,_,valid,loader,train_check,valid_loader=load_data(kind,batch_size=16 if kind=='small' else 4)
    # 모델 생성
    model=build_model(kind,width=width)
    if kind=='partial':
        # fc는 상대적으로 크게, layer4는 작게 학습
        groups=[{'params':model.fc.parameters(),'lr':0.001},{'params':model.layer4.parameters(),'lr':0.0001}]
    # 마지막 분류기(fc)만 학습
    elif kind=='head': groups=model.fc.parameters()
    # SmallCNN 전체 학습
    else: groups=model.parameters()
    # AdamW 옵티마이저 생성
    # - 처음부터 학습하는 SmallCNN은 lr을 크게 사용
    optimizer=torch.optim.AdamW(groups,lr=0.01 if kind=='small' else 0.001,weight_decay=0.0001)
    # epoch별 기록 저장; # 가장 좋은 validation loss 저장; # 학습 시간 측정 시작
    history=[];best=float('inf');start=time.perf_counter()
    for epoch in range(1,epochs+1):
        # 학습 대상 레이어를 설정
        set_training_mode(model,kind)
        # 한 epoch 학습
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x),y).backward();optimizer.step()
        # 학습 데이터 평가
        tl,ta,_,_=evaluate(model,train_check,torch.device('cpu'))
        # 검증 데이터 평가
        vl,va,_,_=evaluate(model,valid_loader,torch.device('cpu'))
        # 결과 기록
        history.append(dict(epoch=epoch,train_loss=tl,valid_loss=vl,train_accuracy=ta,valid_accuracy=va))
        # 가장 좋은 모델 저장
        if vl<best: best=vl;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    # 총 학습 시간 계산
    seconds=time.perf_counter()-start
    # 최적 성능 가중치 복원
    model.load_state_dict(best_state)
    # 최종 validation 성능 측정
    vl,va,_,_=evaluate(model,valid_loader,torch.device('cpu'))
    # 결과 파일 이름
    name=f'{kind}_w{width}'
    # epoch별 학습 기록 저장
    pd.DataFrame(history).to_csv(out/f'history_{name}.csv',index=False)
    # 모델 비교용 정보 저장
    # - trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad) : 실제 학습 가능한 파라미터 수
    # - source_weights='none' if kind=='small' else 'IMAGENET1K_V1') : 사용한 가중치 정보
    rows.append(dict(model=name,trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     image_size=32 if kind=='small' else 224,epochs=epochs,best_epoch=best_epoch,
                     valid_loss=vl,valid_accuracy=va,training_seconds=seconds,
                     source_weights='none' if kind=='small' else 'IMAGENET1K_V1'))
    # 실험 결과 출력
    print(name,'best',best_epoch,'valid loss',round(vl,4),'accuracy',round(va,4))
# 전체 모델 비교표 저장
pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)

print('small32/ResNet224·배치16/4·학습률의 차이가 있으므로 학습 전략 전체의 비교이다.')
