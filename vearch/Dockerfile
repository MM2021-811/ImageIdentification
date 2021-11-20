from vearch/vearch:3.2.5

COPY config.toml /vearch/

WORKDIR /vearch

Entrypoint ["/vearch/bin/vearch", "-conf", "/vearch/config.toml", "-master", "$MASTER_NAME"]
