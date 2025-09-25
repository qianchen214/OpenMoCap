FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg=7:3.4.11-0ubuntu0.1 && \
    apt-mark hold ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/bin:/usr/local/bin:${PATH}
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
ENV MPLBACKEND=Agg

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "solve_and_render.py", "--data_dir", "./test_cases/sfu", "--out_dir", "./results", "--render"]

