import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import nibabel as nib

with open('./resources/aseg_labels.json', 'r') as aseg_label_path:
    aseg_labels = json.load(aseg_label_path)


def execute_command(command, log=False, silent=True):
    """ 
    Executes a shell command and optionally logs the output to a file.

    Args:
        command (str): The shell command to execute.
        log (str, optional): The file path to log the output. Defaults to False.
        silent (bool, optional): If no log is provided, silence output print. Defaults to False.

    Returns:
        None
    """
    try:
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, encoding='utf-8')

        if log:
            with open(log, 'a') as log_path:
                log_path.write(f'{output}\n\n')
        elif not silent:
            print(output,'\n')
            
    except subprocess.CalledProcessError as e:
        if log:
            with open(log, 'a') as log:
                log.write(f'Error running: \n{command}\n\n')
                log.write(f'{e.output}\n')
        else:
            print(output,'\n')

def extract_label(recon_all_dir, label_index, output_file):
    """Extract a specific label from aseg file and save it as a binary volume."""
    input_file = f"{recon_all_dir}/mri/aseg.mgz"
    
    if type(label_index) == list:
        join_niftis = []
        for label in label_index:
            tmp_file = output_file.replace(output_file.split('/')[-1], label + '.nii.gz')
            command = f"mri_binarize --i {input_file} --match {label} --o {tmp_file}"
            execute_command(command)
            join_niftis.append(tmp_file)

        command = f"fslmaths {join_niftis[0]} -add " + " -add ".join(join_niftis[1:]) + f" {output_file}"
        execute_command(command)

    else:
        command = f"mri_binarize --i {input_file} --match {label_index} --o {output_file}"
        execute_command(command)
            
    command = f"fslmaths {output_file} -mul 1 {output_file}"
    execute_command(command)

def nii2obj(input_nifti, output_object):
    """Use nii2mesh package to convert segmentation nifti into surface"""
    command = f"nii2mesh {input_nifti} -l 0 {output_object}"
    execute_command(command)

def convert_aseg_to_surface(recon_all_dir, output_dir, aseg_label, label_index):
    os.makedirs(f"{output_dir}/tmp", exist_ok=True)
    input_path = f"{recon_all_dir}/mri/aseg.mgz"
    nifti_output = f"{output_dir}/tmp/{aseg_label}.nii.gz"
    output_object = f"{output_dir}/{aseg_label}.obj"

    extract_label(recon_all_dir, label_index, nifti_output)
    nii2obj(nifti_output, output_object)
    
    shutil.rmtree(f'{output_dir}/tmp')