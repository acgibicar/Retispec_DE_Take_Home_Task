'''
Created on 2022-09-10
Author: Adam Gibicar
'''

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from datetime import date

class PatientLoader:
    """Patient Loader Class to load patient data and images from multiple sources
    Usage:
    patient = PatientLoader('dataset_id', 'patient_id')
    dataset_id - str, datasource folder name
    patient_id - int, representing row index of patient in metadata file
    """
    def __init__(self, dataset_id, patient_id):
        """Construct patient object"""
        self.dataset_id = dataset_id
        self.patient_id = patient_id
        #run load_metadata_dictionary on object creation
        self.load_metadata_dictionary()

    def load_metadata_dictionary(self):
        """
        Function to generate metadata object (dictionary) from datasource files

        Parameters
        ----------

        Returns
        -------

        Examples
        --------
        patient.load_metadata_dictionary()
        """
        self.patient_metadata = {}

        files = glob.glob(self.dataset_id + '/*.xlsx') + glob.glob(self.dataset_id + '/*.csv')

        if len(files) == 1:
            self.metadata_fname = files[0]

        elif len(files) > 1:
            print("Multiple metadata files found.")
            print("Please select a file to load:")
            for i, file in enumerate(files):
                print(str(i) + ": " + file)
            self.metadata_fname = files[int(input())]

        else:
            print("No metadata files found.")
            return

        try:
            self.metadata_table = pd.read_excel(self.metadata_fname)
        except:
            self.metadata_table = pd.read_csv(self.metadata_fname)

        data_columns = list(self.metadata_table.columns)
        self.patient_data_row = self.metadata_table.iloc[self.patient_id, :]

        for col in data_columns:

            if re.search(r'\bid', col.lower()):
                self.patient_metadata["patient_id"] = self.patient_data_row[col]
            else:
                self.patient_metadata["patient_id"] = str(self.patient_id)
            if re.search(r'\bsex', col.lower()):
                self.patient_metadata["sex"] = self.patient_data_row[col]
            if re.search(r'\bage', col.lower()):
                self.patient_metadata["age"] = self.patient_data_row[col]
                self.patient_metadata["year of birth"] = date.today().year - int(self.patient_data_row[col])

        req_keys = ['patient_id', 'sex', 'age', 'year of birth']
        for req_key in req_keys:
            if req_key in self.patient_metadata.keys():
                continue
            else:
                self.patient_metadata[req_key] = "N/A"
        print("----------------------------------")
        print("Label Keyword Search:")
        self.label_data = self._find_labels()
        print("Labels found: ", self.label_data)
        print("----------------------------------")
        print("Image Search:")
        self.image_ids, self.image_pointers = self._find_images()
        print("Image IDs: ", self.image_ids)
        print("Image pointers: ", self.image_pointers)

        data_dict = {}
        for i in range(len(self.image_ids)):
            data_dict[self.image_ids[i]] = {'data_pointer': self.image_pointers[i], 'data_labels': [self.label_data[i]]}
        self.patient_metadata["data"] = data_dict
        self.patient_metadata["patient_labels"] = self.label_data

    def _find_labels(self):
        """
        Internal function to search for labels in datasource files

        Parameters
        ----------

        Returns
        -------
        label_data : list of labels

        Examples
        --------
        self.label_data = self._find_labels()
        """

        label_dict = {"Normal": "LBL_NORMAL", "Drusen": "LBL_DIABETIC", "Diabetic Retinopathy": "LBL_DIABETIC",
                      "Retinopathy": "LBL_DIABETIC", "Glaucoma": "LBL_GLAUCOMA", "Cataract": "LBL_CATARACT",
                      "Age related Macular Degeneration": "LBL_AMD",
                      "Pathological Myopia": "LBL_MYOPIA", "Other diseases/abnormalities": "LBL_OTHER"}
        label_keys = label_dict.keys()
        data_columns = list(self.metadata_table.columns)
        label_data_type1 = []
        label_data_type2 = []
        for col in data_columns:
            cell_data = self.patient_data_row[col]
            for label in label_keys:
                if re.search(r'\b' + label.lower(), col.lower()):
                    print("Found label in column header:")
                    print(label + " found in " + col)
                    if cell_data > 0:
                        label_data_type1.append(label_dict[label])
                    else:
                        label_data_type1.append("LBL_NORMAL")
                else:
                    if type(cell_data) == str:
                            if label.lower() in cell_data.lower():
                                label_data_type2.append(label_dict[label])
        if len(label_data_type2) < 2:
            label_data_type2.append("LBL_OTHER")

        if len(label_data_type1) == 0:
            label_data = label_data_type2
        else:
            label_data = label_data_type1

        return label_data

    def _find_images(self):
        """
        Internal function to search for labels in datasource files

        Parameters
        ----------

        Returns
        -------
        image_ids : list of image IDs
        image_pointers : list of paths to patient's images

        Examples
        --------
        self.image_ids, self.image_pointers = self._find_images()
        """
        #find image folder
        data_dir = os.listdir(self.dataset_id + '/')
        for file in data_dir:
            file_split = os.path.splitext(file)
            if file_split[1] == '':
                image_dir = file_split[0] + '/'

        image_files = os.listdir(self.dataset_id + '/' + image_dir)

        data_columns = list(self.metadata_table.columns)
        image_ids = []
        image_pointers = []
        #find matching file names in dataframe
        for col in data_columns:
            cell_data = self.patient_data_row[col]
            if type(cell_data) == str:
                for file in image_files:
                    if cell_data == file:
                        image_ids.append(file)
                        image_pointers.append(self.dataset_id + '/' + image_dir + file)

        return image_ids, image_pointers

    def list_patient_images(self):
        """
        Function to return patient image IDs

        Parameters
        ----------

        Returns
        -------
        self.image_ids : list of image IDs

        Examples
        --------
        patient_image_ids = patient.list_patient_images()
        """
        try:
            print("Image IDs for patient: ", self.patient_id)
            print(self.image_ids)
            return self.image_ids
        except:
            print("No patient IDs found.")
            return -1
    def load_patient_image(self, patient_image_id, plot_flag = False):
        """
        Function to return patient image IDs

        Parameters
        ----------
        patient_image_id : str containing image ID
        plot_flag : bool, optional
                    toggle plotting the selected image

        Returns
        -------
        image_data : ndarray of image data

        Examples
        --------
        patient_image = patient.load_patient_image(patient_image_ids[0], plot_flag= True)
        """
        try:
            print("----------------------------------")
            print("Loading image: ", patient_image_id)
            image_path = self.patient_metadata['data'][patient_image_id]['data_pointer']

            image_data = imageio.imread(image_path)
            image_data = np.array(image_data)

            if plot_flag:
                plt.imshow(image_data)
                plt.show()

            return image_data

        except:
            print("No image found for patient.")
            return -1

    def check_patient_label(self, patient_image_id):
        """
        Filter function to check if patient image is Normal or Abnormal

        Parameters
        ----------
        patient_image_id : str containing image ID

        Returns
        -------
        normal_flag : bool
                    True - patient label is Normal
                    False - patient label is Abnormal

        Examples
        --------
        check_label = patient.check_patient_label(patient_image_ids[0])
        """
        print("----------------------------------")
        print("Loading patient label for: ", patient_image_id)
        image_labels = self.patient_metadata['data'][patient_image_id]['data_labels']
        label = image_labels[0]
        print(patient_image_id + " ~ " + label)
        if "normal" in label.lower():
            normal_flag = True
        else:
            normal_flag = False

        return normal_flag

    def analyze_class_distribution(self, plot_flag = True):
        """
        Function to analyze class distribution of entire datasource

        Parameters
        ----------
        plot_flag : bool, optional, default set to True
                    toggle plotting the selected image

        Returns
        -------

        Examples
        --------
        patient.analyze_class_distribution()
        """
        normal_count = 0
        abnormal_count = 0
        img_count = 0

        for i in range(len(self.metadata_table)):
            patient = PatientLoader(dataset_id= self.dataset_id, patient_id= i)
            images_ids = patient.list_patient_images()
            for img in images_ids:
                img_count += 1
                check_label = patient.check_patient_label(img)
                if check_label:
                    normal_count += 1
                else:
                    abnormal_count += 1
        print("----------------------------------")
        print("Label Summary: ", self.dataset_id)
        print("Number of patients = ", len(self.metadata_table))
        print("Number of images = ", img_count)
        print("% of Normal images = ", round(normal_count/img_count, 2))
        print("% of Abnormal images = ", round(abnormal_count/img_count, 2))

        if plot_flag:
            counts = [normal_count, abnormal_count]
            ax = sns.barplot(x = ["Normal", "Abnormal"], y = counts)
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 0.1, counts[i], ha="center")
            plt.xlabel("Label")
            plt.ylabel("Count")
            plt.title("Distribution of Labels Per Image in " + self.dataset_id)
            plt.ylim([0, max(counts) + 1])
            plt.show()

if __name__ == '__main__':
    patient = PatientLoader(dataset_id= 'datasource_1', patient_id= 3)
    print(patient.patient_metadata)

    #Return image IDs
    patient_image_ids = patient.list_patient_images()

    #Return image data using stored path
    patient_image = patient.load_patient_image(patient_image_ids[0], plot_flag= True)

    print("Patient image shape: ", patient_image.shape)

    #Filter function to check if patient has Normal or Abnormal fundus image
    check_label = patient.check_patient_label(patient_image_ids[0])
    if check_label:
        print("Patient image is Normal.")
    else:
        print("Patient image is Abnormal.")

    #Analyze distribution of labels
    patient.analyze_class_distribution()
