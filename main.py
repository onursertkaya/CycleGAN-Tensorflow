#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import cv2 as cv

import os
import time

from network import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

np.set_printoptions(precision=4, threshold=2048,linewidth=500)

def critGAN(prediction, maximize):
    if maximize:
        # force prediction to approach one
        loss = tf.reduce_mean( tf.square( tf.subtract( prediction, tf.ones_like(prediction) ) ) )
    else:
        # force prediction to approach zero
        loss = tf.reduce_mean( tf.square( prediction ) )
    return loss

def critL1(input_1, input_2):
    return tf.reduce_mean( tf.abs( tf.subtract(input_1, input_2) ) )

def main():
    # Prepare Set
    images_dir_path = input('Please provide a dataset path (which contains trainA, trainB, testA, testB)\n')
    
    if images_dir_path[-1] != '/':
      images_dir_path += '/'


    cyclegan_base_path = input('Please provide a base path for program to save checkpoints, generated images and model backups\n')
    
    if cyclegan_base_path[-1] != '/':
      cyclegan_base_path += '/'

    filenames_X = os.listdir(images_dir_path + 'trainA/')
    filenames_Y = os.listdir(images_dir_path + 'trainB/')

    full_paths_of_X = [(images_dir_path + 'trainA/' + filename) for filename in filenames_X]
    full_paths_of_Y = [(images_dir_path + 'trainB/' + filename) for filename in filenames_Y]

    # Prepare Hyperparams
    batch_size = 1
    epochs = 200
    image_size = 256
    lambda_cycle_loss = 10.0
    lambda_identity_loss = 5.0
    dataset_size_X = len(full_paths_of_X)
    dataset_size_Y = len(full_paths_of_Y)
    print('dataset sizes:', dataset_size_X, dataset_size_Y)


    # Initialize epoch and iter count
    most_recent_epoch = tf.Variable(0, trainable=False, name='epoch', dtype=tf.int32)
    most_recent_iter =  tf.Variable(0, trainable=False, name='iter', dtype=tf.int32)
    iter_duration = tf.Variable(0, trainable=False, name='iter_duration', dtype=tf.int32)
    #===========================================================================
    # Placeholders

    lr_d = tf.placeholder(tf.float32, shape=[], name='learning_rate_d')
    lr_g = tf.placeholder(tf.float32, shape=[], name='learning_rate_g')

    # Generator Inputs
    real_X = tf.placeholder(tf.float32, shape=[batch_size,image_size,image_size,3], name='real_X')
    real_Y = tf.placeholder(tf.float32, shape=[batch_size,image_size,image_size,3], name='real_Y')

    # Discriminator Inputs
    fake_X = tf.placeholder(tf.float32, shape=[batch_size,image_size,image_size,3], name='fake_X')
    fake_Y = tf.placeholder(tf.float32, shape=[batch_size,image_size,image_size,3], name='fake_Y')

    #===========================================================================
    # Generator Loss
    fake_Y = generator(real_X, 'gen_X_to_Y', reuse=False) # G(x) aka Y'
    fake_X = generator(real_Y, 'gen_Y_to_X', reuse=False) # F(y) aka X'

    pred_fake_Y_inner = discriminator(fake_Y, 'disc_Y', reuse=False) # Dy(G(x)) aka Dy(Y')
    pred_fake_X_inner = discriminator(fake_X, 'disc_X', reuse=False) # Dx(F(y)) aka Dx(X')

    # Generator Tries to maximize the probability of discriminators' false prediction
    loss_G = critGAN(pred_fake_Y_inner, True)
    loss_F = critGAN(pred_fake_X_inner, True)

    # Cycle Loss
    rec_X = generator(fake_Y, 'gen_Y_to_X', reuse=True) # F(G(x)) aka F(Y') aka X''
    rec_Y = generator(fake_X, 'gen_X_to_Y', reuse=True) # G(F(y)) aka G(X') aka Y''

    loss_cycle_X = lambda_cycle_loss * critL1(rec_X, real_X) # |X - X''| -> 0 (force cyclicX ~= X)
    loss_cycle_Y = lambda_cycle_loss * critL1(rec_Y, real_Y) # |Y - Y''| -> 0

    # Identity Loss
    loss_idt_X = lambda_identity_loss * critL1(real_X, fake_Y)
    loss_idt_Y = lambda_identity_loss * critL1(real_Y, fake_X)

    generative_loss = loss_G + loss_F + loss_cycle_X + loss_cycle_Y + loss_idt_X + loss_idt_Y
    #===========================================================================
    # Discriminator

    pred_real_X = discriminator(real_X, 'disc_X', reuse=True)
    pred_fake_X = discriminator(fake_X, 'disc_X', reuse=True)

    loss_Dx_real = critGAN(pred_real_X, True)
    loss_Dx_fake = critGAN(pred_fake_X, False)



    pred_real_Y = discriminator(real_Y, 'disc_Y', reuse=True)
    pred_fake_Y = discriminator(fake_Y, 'disc_Y', reuse=True)

    loss_Dy_real = critGAN(pred_real_Y, True)
    loss_Dy_fake = critGAN(pred_fake_Y, False)

    loss_Dx = ( loss_Dx_real + loss_Dx_fake ) / 2
    loss_Dy = ( loss_Dy_real + loss_Dy_fake ) / 2

    #===========================================================================
    t_var = tf.trainable_variables()
    dx_vars = [var for var in t_var if 'disc_X' in var.name]
    dy_vars = [var for var in t_var if 'disc_Y' in var.name]
    g_vars = [var for var in t_var if 'gen'  in var.name]

    # print('\n\n\n')
    # for varname in d_vars: print( varname )
    # print('\n\n\n')
    # for varname in g_vars: print( varname )

    # Solvers
    GF_opt = tf.train.AdamOptimizer(lr_g, beta1=0.5).minimize(generative_loss, var_list=g_vars)
    Dx_opt = tf.train.AdamOptimizer(lr_d, beta1=0.5).minimize(loss_Dx, var_list=dx_vars)
    Dy_opt = tf.train.AdamOptimizer(lr_d, beta1=0.5).minimize(loss_Dy, var_list=dy_vars)

    user_decision = input('\n\n[t]rain from scratch.\n[c]ontinue training ?\n[g]enerate image ?\n')
    if not (user_decision == 't' or user_decision == 'c' or user_decision == 'g'):
        print('invalid option, exiting...')
        exit()
    exper_name = input('give me the experiment name\n')

    with tf.Session() as ses:
        # Summaries for Tensorboard
        tf.summary.scalar('disc/loss_dx', loss_Dx)
        tf.summary.scalar('disc/loss_dy', loss_Dy)
        tf.summary.scalar('gen/loss_G', loss_G)
        tf.summary.scalar('gen/loss_F', loss_F)
        tf.summary.scalar('cycl/loss_cyclic_X', loss_cycle_X)
        tf.summary.scalar('cycl/loss_cyclic_Y', loss_cycle_Y)
        tf.summary.scalar('idt/loss_idt_X', loss_idt_X)
        tf.summary.scalar('idt/loss_idt_Y', loss_idt_Y)
        tf.summary.scalar('total/total_gen', generative_loss)
        tf.summary.scalar('info/iter_duration_ms', iter_duration)

        merged = tf.summary.merge_all()

        # Log Writer
        writer = tf.summary.FileWriter( cyclegan_base_path + 'logs/{}'.format(exper_name), ses.graph)

        # Model Saver for checkpoints
        model_saver = tf.train.Saver()

        # Init Variables
        ses.run( tf.global_variables_initializer() )

        prev_epoch_iter = 0
        prev_image_iter = 0

        if user_decision == 'c':
            model_saver.restore(ses, cyclegan_base_path + "checkpoints/{}/{}.ckpt".format(exper_name,exper_name))
            print('model succesfully loaded...')
            prev_epoch_iter = ses.run(most_recent_epoch)
            prev_image_iter = ses.run(most_recent_iter)
            print('\n\nfrom previous training; epoch =',prev_epoch_iter,' iter =',prev_image_iter,'\n\n')

        if user_decision == 'g':
            user_decision = input('[f]ile or [d]ir?')
            with tf.Session() as ses:
                model_saver = tf.train.Saver()
                model_saver.restore(ses, cyclegan_base_path+"checkpoints/{}/{}.ckpt".format(exper_name, exper_name))
                print('model succesfully loaded...')
                if user_decision == 'd':
                    image_path_A = input('give me the image dir for A')
                    image_path_B = input('give me the image dir for B')

                print('generating image(s)...')
                A = [image_path_A + img for img in os.listdir(image_path_A)]
                B = [image_path_B + img for img in os.listdir(image_path_B)]

                iterator = 0
                for img in A:
                    input_tensor = np.zeros([1,256,256,3])
                    input_tensor[0] = (cv.imread(img))/127.5 - 1.0
                    gen_Y = ses.run(fake_Y, feed_dict={ real_X: input_tensor })
                    #print(gen_Y)
                    #input('pause')
                    cv.imwrite(cyclegan_base_path+'generated_images/testA/gen_y_sample_{}.png'.format(iterator), np.round(127.5*(1+gen_Y[0])))
                    iterator += 1
                iterator=0
                for img in B:
                    input_tensor = np.zeros([1,256,256,3])
                    input_tensor[0] = (cv.imread(img))/127.5 - 1.0
                    gen_X = ses.run(fake_X, feed_dict={ real_Y: input_tensor })
                    #print(gen_Y)
                    #input('pause')
                    cv.imwrite(cyclegan_base_path+'generated_images/testB/gen_x_sample_{}.png'.format(iterator), np.round(127.5*(1+gen_X[0])))
                    iterator += 1
                print('generating done...')
            exit()


        pool_X = []
        pool_Y = []
        epoch_adjusted_lr = 2e-4

        for ep_iter in range(prev_epoch_iter, epochs):
            epoch_adjusted_lr = (epoch_adjusted_lr-2e-6) if ep_iter >= 100 else 2e-4
            for image_iter in range(prev_image_iter, dataset_size_X):
                iter_start = time.time()

                A = sample_a_batch(full_paths_of_X, batch_size)
                B = sample_a_batch(full_paths_of_Y, batch_size)

                # Generate Fake X and Y, update Generators' Parameters
                gen_Y, gen_X, gen_loss = ses.run([fake_Y, fake_X, GF_opt],feed_dict={real_X: A, real_Y: B, lr_g: epoch_adjusted_lr})

                # Image Pool
                if(len(pool_X) < 50):
                    pool_X.append(gen_X)
                    pool_Y.append(gen_Y)
                    sampled_X = gen_X
                    sampled_Y = gen_Y

                if(len(pool_X) == 50):
                    p = np.random.rand()
                    if p>0.5:
                        idx = np.random.random_integers(0,49)
                        sampled_X = np.copy( pool_X[idx] )
                        pool_X[idx] = gen_X

                        idx = np.random.random_integers(0,49)
                        sampled_Y = np.copy( pool_Y[idx] )
                        pool_Y[idx] = gen_Y
                    else:
                        sampled_X = gen_X
                        sampled_Y = gen_Y

                # Update Disciminators's Parameters with most recently generated 50 images
                ses.run([Dx_opt, Dy_opt], feed_dict={real_X: A, fake_X: sampled_X, real_Y: B, fake_Y: sampled_Y, lr_d: epoch_adjusted_lr})

                if(image_iter % 50 == 0):
                    # Write summary to disk
                    merged_summary = ses.run(merged,feed_dict={real_X: A, real_Y: B, fake_X: gen_X, fake_Y: gen_Y})
                    writer.add_summary(merged_summary,(dataset_size_X*ep_iter + image_iter))
                    print('epoch[{:3d}/{:3d}] | iter[{:4d}/{:4d}] | lr: {:3f} | {}'.format(ep_iter,epochs,image_iter, dataset_size_X, epoch_adjusted_lr,  time.ctime()) )


                if(image_iter == 0):
                    # Save model to disk
                    ses.run( tf.assign(most_recent_epoch, ep_iter) )
                    ses.run( tf.assign(most_recent_iter, image_iter) )

                    model_saver.save(ses, cyclegan_base_path+"checkpoints/{}/{}.ckpt".format(exper_name,exper_name))

                    # Sample some images and write them to disk
                    [gen_Y, regen_X] = ses.run([fake_Y,rec_X], feed_dict={real_X: A})
                    cv.imwrite(cyclegan_base_path+'generated_images/AB/e{}_regnX.png'.format(ep_iter),  np.round(127.5*( 1 + regen_X[0]  ) ) )
                    cv.imwrite(cyclegan_base_path+'generated_images/AB/e{}_fakeY.png'.format(ep_iter),  np.round(127.5*( 1 + gen_Y[0]  ) ) )
                    cv.imwrite(cyclegan_base_path+'generated_images/AB/e{}_origX.png'.format(ep_iter),  np.round(127.5*( 1 + A[0] ) ) )

                    [gen_X, regen_Y] = ses.run([fake_X,rec_Y], feed_dict={real_Y: B})
                    cv.imwrite(cyclegan_base_path+'generated_images/BA/e{}_regnY.png'.format(ep_iter),  np.round(127.5*( 1 + regen_Y[0]  ) ) )
                    cv.imwrite(cyclegan_base_path+'generated_images/BA/e{}_fakeX.png'.format(ep_iter),  np.round(127.5*( 1 + gen_X[0]  ) ) )
                    cv.imwrite(cyclegan_base_path+'generated_images/BA/e{}_origY.png'.format(ep_iter),  np.round(127.5*( 1 + B[0] ) ) )
                iter_end = time.time()
                ses.run( tf.assign( iter_duration, int(round(1000*(iter_end - iter_start))) ) )

            prev_image_iter = 0


if __name__ == '__main__':
    main()

