FROM python:3.9-slim-bullseye AS builder
RUN apt-get update && apt-get install -y python3-dev gcc libc-dev libffi-dev
WORKDIR /app
COPY . .
RUN python3 -m venv /app
ENV PATH=/app/bin:$PATH
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install wheel
RUN pip install -r requirements.txt

FROM python:3.9-slim-bullseye
# RUN apk add libc6-compat
# RUN apk upgrade --no-cache && apk add --no-cache libgcc libstdc++ ncurses-libs
ENV PATH=/app/bin:$PATH
WORKDIR /app
COPY --from=builder /app /app/
#COPY nltk_data /nltk_data
#ENV NKTL_DATA=/nltk_data
EXPOSE 8000
WORKDIR /app
ENTRYPOINT ["gunicorn","-b","0.0.0.0:8000","--access-logfile","-","--error-logfile","-","--timeout","120"]
CMD ["app:app"]
