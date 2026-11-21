ARG LUMAGRID_DEPENDENCY_IMAGE=python:3.11-slim
FROM ${LUMAGRID_DEPENDENCY_IMAGE} AS dependencies
USER root
WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 LUMAGRID_DATA=/var/lib/lumagrid OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /opt/lumagrid
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY lumagrid lumagrid
COPY static static
RUN useradd -U -u 11506 lumagrid && install -d -o lumagrid -g lumagrid /var/lib/lumagrid
USER lumagrid
EXPOSE 8000
CMD ["python", "-m", "lumagrid"]
