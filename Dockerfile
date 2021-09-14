# Dockerfile's arguments:
# tag: Tensorflow Image tag (GPU: 2.1.0-gpu-py3, CPU: 2.1.0-py3)
ARG image=tensorflow/tensorflow
ARG tag=2.1.0-gpu-py3
# 2.4.1

# Base image
FROM ${image}:${tag}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install some essentials
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential=\* \
        cmake=\* \
        g++-4.8=\* \
        git=\* \
        curl=\* \
        wget=\* \
        ca-certificates=\* && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Install R from r-project repos
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    	software-properties-common=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
	r-base=\* \
	r-base-dev=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure Java and R
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt-get update -y && \
    apt-get install -y  --no-install-recommends \
         libgit2-dev=\* \
         libcurl4-gnutls-dev=\* \
         libicu-dev=\* \
         libbz2-dev=\* \
         libnetcdf-dev=\* \
         libnetcdff-dev=\* \
         libssl-dev=\* \
         libssh2-1-dev=\* \
         libxml2-dev=\* \
         libgit2-dev=\* && \
    R -e "install.packages('devtools')" && \
    apt-get install -y --no-install-recommends \
         default-jre=\* \
         default-jdk=\* && \
    R CMD javareconf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl repo.data.kit.edu/key.pgp | apt-key add - && \
    add-apt-repository "deb http://repo.data.kit.edu/ubuntu/$(lsb_release -sr) ./" && \
    apt-get install -y --no-install-recommends \
	liblzma-dev=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install virtual framebuffer X11 server 
RUN apt-get update -y && apt-get install -y --no-install-recommends \
	xvfb=\* \
	xauth=\* \
	xfonts-base=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install climate4R 
RUN R -e "library('devtools'); install_github('SantanderMetGroup/loadeR.java')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/climate4R.UDG')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/loadeR')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/transformeR')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/downscaleR')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/VALUE')"
RUN R -e "library('devtools'); install_github('SantanderMetGRoup/climate4R.value')"
RUN R -e "library('devtools'); install_github('SantanderMetGRoup/downscaleR.keras@devel')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/loadeR.2nc')"
RUN R -e "install.packages(c('Matrix', 'RcppEigen', 'spam'))"

# Install geprocessoR and gdal, rgdal and proj dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        gdal-bin=\* \
        proj-bin=\* \
        libgdal-dev=\* \
        libproj-dev=\* \
        libudunits2-dev=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages('rgdal')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/geoprocessoR')"

RUN R -e "install.packages('udunits2')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/convertR')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/climate4R.indices')"

# Install Keras and Tensorflow on R
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN R -e "install.packages('keras')"
RUN R -e "reticulate::use_python('/usr/bin/python', required = TRUE)"

# Install graphical packages
RUN R -e "install.packages('gridExtra')"
RUN R -e "install.packages('RColorBrewer')"
RUN R -e "library('devtools'); install_github('SantanderMetGroup/visualizeR')"

# Install Pytorch and related
RUN pip install torch torchvision torchaudio
RUN pip install rpy2 scikit-learn matplotlib netCDF4 
RUN pip install jupyter

# Install the R kernel for Jupyter Notebook
RUN R -e "install.packages('IRkernel'); IRkernel::installspec()"

# Install wand for showing PDF Images in Jupyter Notebooks
RUN pip install wand
RUN apt-get update -y && apt-get install -y \
        libmagickwand-dev=\* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm /etc/ImageMagick-6/policy.xml

WORKDIR "/experiment/"
