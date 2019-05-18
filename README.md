# Scene parsing

End to end semnatic segmantation using deeplab v3 trained on ade20k

## Installation

using Docker

```shell
docker build -t scene-parsing:latest .
```

without Docker

```shell
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Running

with Docker

```shell
docker run -d -p 3030:3030 scene-parsing
```

without docker

```
python api.py
```


## Results

```bash
mIOU: 45.65% (val)
Pixel-wise Accuracy: 82.52% (val)
File Size: 460 mb
image size: 512 * 512
```


