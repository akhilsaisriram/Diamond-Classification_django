from openai import OpenAI
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from xgboost import XGBClassifier
import cv2
from django.http import JsonResponse
import numpy as np
from rest_framework.parsers import MultiPartParser
import json
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
import torch
import pickle
class ImageUploadView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request, *args, **kwargs):


        file_obj = request.FILES.get('image')
        print(file_obj)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_modela.pkl')
        model_patha = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'label_encoder.pkl')

        # Load the saved model using joblib
        loaded_model = joblib.load(model_path)


        # Load the LabelEncoder if you have it saved along with the model
        # Replace 'label_encoder_path' with the actual path to your LabelEncoder file
        label_encoder = joblib.load(model_patha)  # Adjust the filename and path
        try:
                # Read the uploaded image using OpenCV
                img = cv2.imdecode(np.frombuffer(file_obj.read(), np.uint8), cv2.IMREAD_COLOR)

                # Convert the image to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Calculate color histogram features
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist_features = hist.flatten()

                # Calculate Hu Moments as shape features
                moments = cv2.moments(gray_img)
                hu_moments = cv2.HuMoments(cv2.moments(gray_img)).flatten()

                # Concatenate features into a single feature vector
                feature_vector = np.concatenate((hist_features, hu_moments))

                # Perform prediction using the loaded model
                # Make sure your model expects the input in the correct format (e.g., feature vector)
                prediction = loaded_model.predict([feature_vector])  # Replace this with your model prediction logic
                print(prediction)

                # Convert the prediction to a human-readable format
                # predicted_class = str(prediction[0])  # Assuming prediction is a class label
                predicted_class = label_encoder.inverse_transform(prediction)[0]

                return JsonResponse({'predicted_class': predicted_class})

                # return JsonResponse({'predicted_class': predicted_class})

        except Exception as e:
            return JsonResponse({'error': str(e)})


# import torchvision.models as models
# import torch.nn as nn
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision.models import resnet18


# class ImageUploadView(APIView):
#     parser_classes = (MultiPartParser,)

#     def post(self, request, *args, **kwargs):
#         # Get the uploaded image file
#         file_obj = request.FILES.get('image')

#         # Read image data from file object
#         img_bytes = file_obj.read()
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Convert BGR to RGB
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Normalize pixel values
#         img_normalized = img_rgb / 255.0

#         # Convert to PyTorch tensor
#         img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(
#             2, 0, 1)  # Channels first (C, H, W)

#         # Print the shape of img_tensor
#         print(img_tensor.shape)

#         # Load the model
#         model_path = os.path.join(os.path.dirname(
#             os.path.abspath(__file__)), 'resnet_with_attention.pth')
#         try:

#             class SelfAttention(nn.Module):
#                 def __init__(self, in_channels):
#                     super(SelfAttention, self).__init__()
#                     self.query_conv = nn.Conv2d(
#                         in_channels, in_channels//8, kernel_size=1)
#                     self.key_conv = nn.Conv2d(
#                         in_channels, in_channels//8, kernel_size=1)
#                     self.value_conv = nn.Conv2d(
#                         in_channels, in_channels, kernel_size=1)
#                     self.gamma = nn.Parameter(torch.zeros(1))

#                 def forward(self, x):
#                     batch_size, C, width, height = x.size()
#                     proj_query = self.query_conv(x).view(
#                         batch_size, -1, width * height).permute(0, 2, 1)
#                     proj_key = self.key_conv(x).view(
#                         batch_size, -1, width * height)
#                     energy = torch.bmm(proj_query, proj_key)
#                     attention = torch.softmax(energy, dim=-1)
#                     proj_value = self.value_conv(x).view(
#                         batch_size, -1, width * height)

#                     out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#                     out = out.view(batch_size, C, width, height)

#                     out = self.gamma * out + x
#                     return out

#             # Define the model architecture
#             class NETWithAttentionAndClassifier(nn.Module):
#                 def __init__(self, pretrained_model, num_classes, channel):
#                     super(NETWithAttentionAndClassifier, self).__init__()
#                     self.resnet = nn.Sequential(
#                         *list(pretrained_model.children())[:-2])
#                     # Assuming SelfAttention is defined somewhere
#                     self.attention = SelfAttention(channel)
#                     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#                     self.fc = nn.Linear(channel, num_classes)

#                 def forward(self, x):
#                     x = self.resnet(x)
#                     x = self.attention(x)
#                     x = self.avgpool(x)
#                     x = torch.flatten(x, 1)
#                     x = self.fc(x)
#                     return x

#             # Load the model
#             model = NETWithAttentionAndClassifier(models.resnet18(
#                 pretrained=True), num_classes=10, channel=512)
#             # model = models.resnet18(pretrained=True)
#             # Load the state dictionary into the model
#             model.load_state_dict(torch.load(
#                 model_path, map_location=torch.device('cpu')))

#             print("Model loaded successfully.")

#             # Set the model to evaluation mode
#             model.eval()

#             # Perform prediction
#             with torch.no_grad():
#                 outputs = model(img_tensor.unsqueeze(0))  # Add batch dimension
#                 _, predicted = torch.max(outputs, 1)
#                 print(predicted.item())
#                 # return JsonResponse({'predicted_class':'ruby'})

#         except FileNotFoundError:
#             print(f"Model file '{model_path}' not found.")
#             return JsonResponse({'error': 'Model file not found'}, status=500)
#         except Exception as e:
#             print(f"Error loading model: {str(e)}")
#             return JsonResponse({'error': 'Error loading model'}, status=500)

#         return JsonResponse({'predicted_class': predicted.item()})
    

# # class ChatgptView(APIView):
# #     def post(self, request):
# #         input = request.data['input']
# #         openai = OpenAI(
# #             api_key='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',
# #         )
# #         try:
# #             input_text = input
# #             template = f"""
# #             You are chatting with a plant disease information system. Your goal is to clear the user's doubts and provide answers concisely, in less than 200 words.

# #             Begin!

# #             User Input:
# #             {input_text}
# #         """

# #             completion = openai.chat.completions.create(
# #                 model="gpt-3.5-turbo",
# #                 messages=[
# #                     {"role": "system", "content": template},
# #                 ]
# #             )

# #             response = completion.choices[0].message.content
# #             print({"output": response})

# #         except Exception as e:
# #             print({"error": str(e)})

# #         return Response("output",response)

# from rest_framework.response import Response

# class ChatgptView(APIView):
#     def post(self, request):
#         input_text = request.data.get('input', '')  # Using get to handle cases when input is not provided
#         openai = OpenAI(
#             api_key='',
#         )
#         try:
#             template = f"""
#             You are chatting with a plant disease information system. Your goal is to clear the user's doubts and provide answers concisely, in less than 100 words.

#             Begin!

#             User Input:
#             {input_text}
#             """

#             completion = openai.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": template},
#                 ]
#             )

#             response = completion.choices[0].message.content
#             print({"output": response})

#             # Return the response using Response object
#             return Response({"output": response})  # Response data wrapped in a dictionary

#         except Exception as e:
#             print({"error": str(e)})
#             # Return an error response if an exception occurs
#             return Response({"error": str(e)}, status=500)  # Internal Server Error status code
