# Adapted from https://github.com/joonson/syncnet_python/blob/master/SyncNetInstance.py

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from .syncnet import S
from shutil import rmtree
from pathlib import Path
import os
def check_model_and_download(ckpt_path: str, huggingface_model_id: str = "ByteDance/LatentSync-1.5"):
    if not os.path.exists(ckpt_path):
        ckpt_path_obj = Path(ckpt_path)
        subprocess.run(
            [
                "hf",
                "download",
                huggingface_model_id,
                str(Path(*ckpt_path_obj.parts[1:])),
                "--local-dir",
                str(Path(ckpt_path_obj.parts[0])),
            ],
            check=True,
        )

# ==================== Get OFFSET ====================

# Video 25 FPS, Audio 16000HZ


def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    dists = []

    for i in range(0, len(feat1)):

        dists.append(
            torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i : i + win_size, :])
        )

    return dists


# ==================== MAIN DEF ====================


class SyncNetEval(torch.nn.Module):
    def __init__(self, dropout=0, num_layers_in_fc_layers=1024, device="cpu"):
        super().__init__()

        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).to(device)
        self.device = device

    def evaluate(self, video_path, temp_dir="temp", batch_size=20, vshift=15):

        self.__S__.eval()

        # ========== ==========
        # Convert files
        # ========== ==========

        if os.path.exists(temp_dir):
            rmtree(temp_dir)

        os.makedirs(temp_dir)

        # temp_video_path = os.path.join(temp_dir, "temp.mp4")
        # command = f"ffmpeg -loglevel error -nostdin -y -i {video_path} -vf scale='224:224' {temp_video_path}"
        # subprocess.call(command, shell=True)

        command = f"ffmpeg -loglevel error -nostdin -y -i {video_path} -f image2 {os.path.join(temp_dir, '%06d.jpg')}"
        subprocess.run(command, shell=True, check=True, stdout=None)

        command = f"ffmpeg -loglevel error -nostdin -y -i {video_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {os.path.join(temp_dir, 'audio.wav')}"
        subprocess.run(command, shell=True, check=True, stdout=None)

        # ========== ==========
        # Load video
        # ========== ==========

        images = []

        flist = glob.glob(os.path.join(temp_dir, "*.jpg"))
        flist.sort()

        for fname in flist:
            img_input = cv2.imread(fname)
            img_input = cv2.resize(img_input, (224, 224))  # HARD CODED, CHANGE BEFORE RELEASE
            images.append(img_input)

        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(os.path.join(temp_dir, "audio.wav"))
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        # if (float(len(audio)) / 16000) != (float(len(images)) / 25):
        #     print(
        #         "WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."
        #         % (float(len(audio)) / 16000, float(len(images)) / 25)
        #     )

        min_length = min(len(images), math.floor(len(audio) / 640))

        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0, lastframe, batch_size):

            im_batch = [imtv[:, :, vframe : vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
            im_in = torch.cat(im_batch, 0)
            im_out = self.__S__.forward_lip(im_in.to(self.device))
            im_feat.append(im_out.data.cpu())

            cc_batch = [
                cct[:, :, :, vframe * 4 : vframe * 4 + 20] for vframe in range(i, min(lastframe, i + batch_size))
            ]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = self.__S__.forward_aud(cc_in.to(self.device))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        # ========== ==========
        # Compute offset
        # ========== ==========

        dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
        mean_dists = torch.mean(torch.stack(dists, 1), 1)

        min_dist, minidx = torch.min(mean_dists, 0)

        av_offset = vshift - minidx
        conf = torch.median(mean_dists) - min_dist

        fdist = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf = torch.median(mean_dists).numpy() - fdist
        framewise_conf = signal.medfilt(fconf, kernel_size=9)

        # numpy.set_printoptions(formatter={"float": "{: 0.3f}".format})
        rmtree(temp_dir)
        return av_offset.item(), min_dist.item(), conf.item()

    def extract_feature(self, opt, videofile):

        self.__S__.eval()

        # ========== ==========
        # Load video
        # ========== ==========
        cap = cv2.VideoCapture(videofile)

        frame_num = 1
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images) - 4
        im_feat = []

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):

            im_batch = [
                imtv[:, :, vframe : vframe + 5, :, :] for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            im_in = torch.cat(im_batch, 0)
            im_out = self.__S__.forward_lipfeat(im_in.to(self.device))
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)

        # ========== ==========
        # Compute offset
        # ========== ==========

        print("Compute time %.3f sec." % (time.time() - tS))

        return im_feat

    def loadParameters(self, path):
        check_model_and_download(path)
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=True)

        self_state = self.__S__.state_dict()

        for name, param in loaded_state.items():

            self_state[name].copy_(param)
