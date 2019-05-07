# :star: [NEW] :star:
# T2F - 2.0 Teaser (coming soon ...)
<p align="center">
<img src="https://raw.githubusercontent.com/akanimax/T2F/master/figures/T2F_2.0_teaser.jpeg" alt="2.0 Teaser">
</p>

## Please note that all the faces in the above samples are generated ones. The T2F 2.0 will be using MSG-GAN for the image generation module instead of ProGAN. Please refer [link](https://github.com/akanimax/BMSG-GAN) for more info about MSG-GAN. This update to the repository will be comeing soon :+1:.

# T2F
Text-to-Face generation using Deep Learning. This project combines two of the recent architectures <a href="https://arxiv.org/abs/1710.10916"> StackGAN </a> and <a href="https://arxiv.org/abs/1710.10196"> ProGAN </a> for synthesizing faces from textual descriptions.<br>
The project uses <a href="https://arxiv.org/abs/1803.03827"> Face2Text </a> dataset which contains 400 facial images and textual captions for each of them. The data can be obtained by contacting either the **RIVAL** group or the authors of the aforementioned paper.

<h3>Some Examples:</h3>
<img src="https://github.com/akanimax/T2F/blob/master/figures/result.jpeg" alt="Examples">

<h3>Architecture: </h3>
<img src="https://github.com/akanimax/T2F/blob/master/figures/architecture.jpg" alt="Architecture Diagram">
The textual description is encoded into a summary vector using an LSTM network. The summary vector, i.e. <b>Embedding</b> <i>(psy_t)</i> as shown in the diagram is passed through the Conditioning Augmentation block (a single linear layer) to obtain the textual part of the latent vector (uses VAE like reparameterization technique) for the GAN as input. The second part of the latent vector is random gaussian noise. The latent vector so produced is fed to the generator part of the GAN, while the embedding is fed to the final layer of the discriminator for conditional distribution matching. The training of the GAN progresses exactly as mentioned in the ProGAN paper; i.e. layer by layer at increasing spatial resolutions. The new layer is introduced using the fade-in technique to avoid destroying previous learning.

## Running the code:
The code is present in the `implementation/` subdirectory. The implementation is done using the <a href="https://pytorch.org/"> PyTorch</a> framework. So, for running this code, please install `PyTorch version 0.4.0` before continuing.

__Code organization:__ <br>
`configs`: contains the configuration files for training the network. (You can use any one, or create your own) <br>
`data_processing`: package containing data processing and loading modules <br>
`networks`: package contains network implementation <br>
`processed_annotations`: directory stores output of running `process_text_annotations.py` script <br>
`process_text_annotations.py`: processes the captions and stores output in `processed_annotations/` directory. (no need to run this script; the pickle file is included in the repo.) <br>
`train_network.py`: script for running the training the network <br>

__Sample configuration:__

    # All paths to different required data objects
    images_dir: "../data/LFW/lfw"
    processed_text_file: "processed_annotations/processed_text.pkl"
    log_dir: "training_runs/11/losses/"
    sample_dir: "training_runs/11/generated_samples/"
    save_dir: "training_runs/11/saved_models/"

    # Hyperparameters for the Model
    captions_length: 100
    img_dims:
      - 64
      - 64

    # LSTM hyperparameters
    embedding_size: 128
    hidden_size: 256
    num_layers: 3  # number of LSTM cells in the encoder network

    # Conditioning Augmentation hyperparameters
    ca_out_size: 178

    # Pro GAN hyperparameters
    depth: 5
    latent_size: 256
    learning_rate: 0.001
    beta_1: 0
    beta_2: 0
    eps: 0.00000001
    drift: 0.001
    n_critic: 1

    # Training hyperparameters:
    epochs:
      - 160
      - 80
      - 40
      - 20
      - 10
    
    # % of epochs for fading in the new layer
    fade_in_percentage:
      - 85
      - 85
      - 85
      - 85
      - 85

    batch_sizes:
      - 16
      - 16
      - 16
      - 16
      - 16

    num_workers: 3
    feedback_factor: 7  # number of logs generated per epoch
    checkpoint_factor: 2  # save the models after these many epochs
    use_matching_aware_discriminator: True  # use the matching aware discriminator

Use the `requirements.txt` to install all the dependencies for the project. 
    
    $ workon [your virtual environment]
    $ pip install -r requirements.txt

__Sample run:__

    $ mkdir training_runs
    $ mkdir training_runs/generated_samples training_runs/losses training_runs/saved_models
    $ train_network.py --config=configs/11.comf


## Other links:
blog: https://medium.com/@animeshsk3/t2f-text-to-face-generation-using-deep-learning-b3b6ba5a5a93 <br>
training_time_lapse video: https://www.youtube.com/watch?v=NO_l87rPDb8 <br>
ProGAN package (Seperate library): https://github.com/akanimax/pro_gan_pytorch

## #TODO:
1.) Create a simple `demo.py` for running inference on the trained models <br>
