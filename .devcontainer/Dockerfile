FROM continuumio/anaconda3 AS upstream

# Verify OS version is expected one
RUN . /etc/os-release && if [ "${VERSION_CODENAME}" != "bookworm" ]; then exit 1; fi

# Temporary: Upgrade python packages due to mentioned CVEs
# They are installed by the base image (continuumio/anaconda3) which does not have the patch.
COPY .devcontainer/apply_security_patches.sh /tmp/apply_security_patches.sh
RUN chmod +x /tmp/apply_security_patches.sh
RUN /tmp/apply_security_patches.sh

# Reset and copy updated files with updated privs to keep image size down
FROM mcr.microsoft.com/devcontainers/base:1-bookworm

ARG USERNAME=vscode

# Create the conda group and add remote user to the group
RUN groupadd -r conda --gid 900 \ 
    && usermod -aG conda ${USERNAME}

# Copy opt folder, set ownership and group permissions
COPY --chown=:conda --chmod=775 --from=upstream /opt/conda /opt/conda
RUN chmod =2775 /opt/conda

USER root

# Setup conda to mirror contents from https://github.com/ContinuumIO/docker-images/blob/master/anaconda3/debian/Dockerfile
ARG CONTAINER_WORKSPACE_FOLDER
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=${CONTAINER_WORKSPACE_FOLDER}/src
ARG USERNAME=vscode
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        wget \
    && apt-get upgrade -y \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Copy environment.yml (if found) to a temp location so we can update the environment. 
COPY environment.yml* /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# ensure group members can do conda commands in the /opt/conda directory structure
# NOTE: this takes a long time to do, so left out of setup.  Need to run to make sure all files in /opt/conda in in conda group if you
# need to install or manage conda packages from the vscode user
#RUN chmod -R g+rw /opt/conda
#RUN chown -R root:conda /opt/conda
#RUN find /opt/conda -group root -print | xargs chown root:conda

# Enable contrib component in debian apt sources so we can install nvtop
# and other packages
RUN sed -i 's/Components: main/Components: main contrib/g' /etc/apt/sources.list.d/debian.sources

# Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends nvtop

# initialize conda for the regular user
USER vscode

RUN conda init
