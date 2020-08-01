FROM python:3.7.3-slim-stretch

### Stage 0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # git because some Python modules come from git
    git openssh-client \
    # compiler for Python C modules
    g++ make libc6-dev

RUN groupadd -g 1337 hmmconf && \
    useradd -m -d /opt/hmmconf -u 1337 -g hmmconf hmmconf

USER hmmconf

# install Python requirements
ADD requirements.txt /tmp/requirements.txt
ADD --chown=hmmconf:hmmconf ./dev /usr/local/bin/

RUN dev pipi -r /tmp/requirements.txt && \
    # clean up Python modules
    find /opt/hmmconf/.local \
    \( -type d -a -name test -o -name tests \) \
    -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
    -exec rm -rf '{}' \;

### Stage 1

FROM python:3.7.3-slim-stretch

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # to make life easier
    procps neovim less

RUN groupadd -g 1337 hmmconf && \
    useradd -m -d /opt/hmmconf -u 1337 -g hmmconf hmmconf && \
    mkdir /var/log/hmmconf && \
    chown hmmconf:hmmconf /var/log/hmmconf

USER hmmconf
WORKDIR /opt/hmmconf
ENV PATH /opt/hmmconf/.local/bin:$PATH
COPY --chown=hmmconf:hmmconf --from=0 /opt/hmmconf/.local /opt/hmmconf/.local

EXPOSE 8000

ADD --chown=hmmconf:hmmconf ./docker-entrypoint.sh /
ADD --chown=hmmconf:hmmconf ./dev /usr/local/bin/

ENTRYPOINT ["/docker-entrypoint.sh"]
