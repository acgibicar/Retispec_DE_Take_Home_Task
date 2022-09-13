# PatientLoader
PatientLoader is a Python Class to used load patient data and images from multiple sources.

## Installation
```sh
git clone https://github.com/acgibicar/Retispec_DE_Take_Home_Task.git
pip install -r requirements.txt
```

## Usage
```sh
python adam_gibicar_de_takehome_task.py
```

### Create Patient Object and print metadata
```sh
patient = PatientLoader(dataset_id= 'datasource_1', patient_id= 3)
print(patient.patient_metadata)
```
### Return image IDs
```sh
patient_image_ids = patient.list_patient_images()
```
### Return image data using stored path
```sh
patient_image = patient.load_patient_image(patient_image_ids[0], plot_flag= False)
```

### Filter function to check if patient has Normal or Abnormal fundus image
```sh
check_label = patient.check_patient_label(patient_image_ids[0])
if check_label:
    print("Patient image is Normal.")
else:
    print("Patient image is Abnormal.")
```
### Analyze distribution of labels
```sh  
patient.analyze_class_distribution()
```
