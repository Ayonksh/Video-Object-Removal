# Video-Object-Removal

只要给物体画上一个方框，就可以在视频中去除这个物体并修复视频

## 使用步骤

+ 安装依赖

```shell
cd video-object-removal
cd removing
bash make.sh
cd inpainting
bash install.sh
```

+ 下载预训练模型，放在 `pretrained_models/`文件夹中

本项目基于 [SiamMask](https://github.com/foolwood/SiamMask) 和 [Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting) 。预训练模型请参考这两个项目

+ 运行

```bash
python main.py --src demo.mp4
```

## 效果展示

+ 画方框

![](./doc/drawbox.gif)

+ 目标将被去除，修复好的视频保存在 `results` 文件夹中

![](./doc/removing.gif)

![](./doc/skate.gif)