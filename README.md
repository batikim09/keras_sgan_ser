# keras_sgan_ser
This is a project of emotion recognition using KERAS based Semi-Generative Adversarial Networks (SGAN).

The implementation of SGAN is largely affected by:
https://github.com/eriklindernoren/Keras-GAN

Many ideas about feature structures are based on:

Kim, Jaebok, Gwenn Englebienne, Khiet P Truong, and Vanessa Evers. “Deep Temporal Models using Identity Skip-Connections for Speech Emotion Recognition.” In: Proceedings of ACM Multimedia 2017

Kim, Jaebok, Khiet Truong, Gwenn Englebienne, and Vanessa Evers. “Learning spectral-temporal features with 3D CNNs for speech emotion recognition.” In: Proceedings of International Conference on Affective Computing and Intelligent Interaction 2017


The following sections will explain how to install, prepare data, run experiments.

# Installment
Once you clone this repository, all dependent libraries should be installed using pip:
sudo pip install -r requirements.txt

# Tutorials
After installing all the required libraries,
see the script: tutorial.sh

In this script, various experiments will be conducted for sanity checks and better understanding of SGAN.

# Data preparation
Since I can't upload any corpus (or features) here, i assume that you have a data set with(out) labels. For example, in this script, I use the following path of the database:

../../features/ser/stl_vs_mtl/lstm/LSPEC.IMG.100.2d.cc.REC.4cls.av.h5

This data has log-spectrogram features extracted from the RECOLA corpus
(https://diuf.unifr.ch/diva/recola/download.html
F. Ringeval, A. Sonderegger, J. Sauer and D. Lalanne, "Introducing the RECOLA Multimodal Corpus of Remote Collaborative and Affective Interactions", 2nd International Workshop on Emotion Representation, Analysis and Synthesis in Continuous Time and Space (EmoSPACE), in Proc. of IEEE Face & Gestures 2013, Shanghai (China), April 22-26 2013.)

The database has three data: "a_feat", "v_feat", and "label", representing that audio, video(face), and emotional labels(arousal, valence, and etc.). Note that "a_feat" and "v_feat" are just key values, they can be any values as defined in the database.

"a_feat" has the following structure in numpy matrix: (#samples x #long_frames x #channels x #short_frames x #feature dimensions)
For speech, the number of channels is always one. Since our SGAN currently supports only 2D CNNs, we might not need these complex structure, but it is for future extensions to RNN or 3D CNNs soon (See Citations). 

In this data, "a_feat" has a matrix with size of (n_samples x 10 x 1 x 10 x 256)
If your data has longer frames or higher feature dimensions, You can resize it using arguments (r_nrow, r_ncol).

"v_feat" has a shape of (#samples x #frames x #channels x #rows x #columns) to represent raw facial images. The actual size is (n_samples x 1 x 1 x 48 x 48) for the purpose of sanity checks.

"label" has a shape of (#samples x #labels)
In this databae, "label" has arousal, valence, discrete categories, and meta information such as speaker ID, corpus ID, gender, and etc.
You can select which category you want to use as the label using args "-mt" (run "python sgan_mer.py -h" to see the details)

# Running experiments
Open "tutorial.sh"

The first script (1) trains SGAN using both audio and video features. As the number of generators(or discriminators) is 2, we need two configurations, separated by ";", in many args (e.g. -cnn_n_col_D '16,64,128;8,16,32'). 
Run "python sgan_mer.py -h" to see the details.

In (1), the SGAN will have two generators and two discriminators but the discriminators will be merged for classification of labels and regression of validity (real/fake).

# Citations
Please cite the paper in your publications if it helps your research:

@inproceedings{kim2017acmmm,
  title={Deep Temporal Models using Identity Skip-Connections for Speech Emotion Recognition},
  author={Kim, Jaebok and Englebienne, Gwenn and Truong, Khiet P and Evers, Vanessa},
  booktitle={Proceedings of ACM Multimedia},
  pages={To be appeared},
  year={2017}
}

@inproceedings{kim2017acii,
  title={Learning spectral-temporal features with 3D CNNs for speech emotion recognition},
  author={Kim, Jaebok and Truong, Khiet and Englebienne, Gwenn and Evers, Vanessa},
  booktitle={Proceedings of International Conference on Affective Computing and Intelligent Interaction},
  pages={To be appeared},
  year={2017}
}