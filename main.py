import cv2
import torch
import gradio as gr
from PIL import Image
import os
import atexit
import shutil
from hubconf import custom

# Event Callback Function
def yolo_image(im):
    if im is None:
        return None
    
    model.conf = conf
    results = model(im)
    results.render()
    
    return Image.fromarray(results.imgs[0])
# Event Callback Function
def yolo_video(video):
    if video is None:
        return None
    
    # YOLO process
    def yolo(im):
        results = model(im)
        results.render()
        
        return results.imgs[0]
    
    # Delete tmp Video
    def delete_output_video():
        if os.path.exists(tmp_output_video):
            os.remove(tmp_output_video)

    atexit.register(delete_output_video)

    # Open Video
    cap = cv2.VideoCapture(video)

    # VideoWriter setting
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_output_video = 'tmp_processed_output.mp4'

    if width > 1280:
        width = 1280
        height = int(height * (1280/width))
        
    output_vw = cv2.VideoWriter(tmp_output_video, fourcc, 24,(width, height))
    
    try:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            model.conf = conf
            result = yolo(frame)
            output_vw.write(result)

    except Exception as e:
        print("Error Occured: ", str(e))
        # Delete tmp video
        delete_output_video()
    
    finally:
        cap.release()
        output_vw.release()
        
        return tmp_output_video

def slider_callback(value):
    global conf
    conf = value

def load_model(modeldropdown, devicedropdown):
    global model, device
    gr.Warning("Wait for Load Model")
    if torch.cuda.is_available() is not True and devicedropdown != "cpu":
        return gr.Warning("CUDA를 사용할 수 없습니다.")
    
    scrip_dir = os.path.dirname(os.path.abspath(__file__))
    if modeldropdown is None:
        return dropdown.update()
    model_path = scrip_dir + "/weights/" + modeldropdown
    model = custom(model_path, device=devicedropdown)
    
    
    if devicedropdown != "cpu":
        model.half()
    
    model.conf = conf

    return gr.Info(f"Device :{next(model.parameters()).device}")

def upload_save(file):
    
    file_name = os.path.basename(file.name)
    save_path = os.path.dirname(os.path.abspath(__file__)) + "/weights/" + file_name
    try:
        shutil.copy(file.name, save_path)
    except Exception as e:
        print("Error!!", e)
        return uploadbtn.update(label="Error Occured! Press F5")


    updated_list = [f for f in os.listdir("weights") if os.path.isfile(os.path.join("weights", f))]
    return dropdown.update(choices=updated_list)

def delete_callback(file_):
    
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/weights/" + file_
    os.remove(file_path)


    updated_list = [f for f in os.listdir("weights") if os.path.isfile(os.path.join("weights", f))]

    return dropdown.update(choices=updated_list, value=None)

def refresh():
    updated_list = [f for f in os.listdir("weights") if os.path.isfile(os.path.join("weights", f))]
    print("REFRESHING")
    return dropdown.update(choices=updated_list)

# Initial Setting
scrip_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(scrip_dir)

if not os.path.exists("weights"):
    os.makedirs("weights")

conf = 0.25
model_list = [f for f in os.listdir("weights") if os.path.isfile(os.path.join("weights", f))]
model = None
device = None

# Gradio Blocks Setting
Knowgyu = gr.Blocks(theme="Soft", title="Model_Test")
with Knowgyu:
    gr.Markdown('''
                # Object Detection Model Test
                ### 사용법 :
                - 모델 선택 → 이미지(혹은 동영상) 업로드 → 'Run' 버튼 클릭
                - 'Upload model' 버튼을 통해 모델을 추가할 수 있습니다.
                ''')

    with gr.Tab("Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image(label="Result")

        with gr.Row():
            slider_input1 = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence Threshold", 
                                    interactive=True, container=True)
            image_button = gr.Button("Run")
            
            
    with gr.Tab("Video"):
        with gr.Row():
            video_input = gr.Video()
            video_output = gr.Video(label="Result")

        with gr.Row():
            slider_input2 = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence Threshold",
                                    interactive=True, container=True)
            video_button = gr.Button("Run")

    with gr.Row():
        with gr.Column():
            dropdown = gr.Dropdown(label="Select Model", choices=model_list, 
                                   container=True,interactive=True)
            devicedropdown = gr.Dropdown(label="Select Device", choices=["cpu","0","1"], value="0")
        with gr.Column():
            uploadbtn = gr.UploadButton(label="Upload model", type='file'
                                        , file_types=[".pt"])
            deletebtn = gr.Button(value="Delete selected model", interactive=True)
            

    ## Event Callback Functions ##
    ##############################
    slider_input1.change(fn = slider_callback, inputs=slider_input1)
    slider_input2.change(fn = slider_callback, inputs=slider_input2)
    dropdown.change(fn = load_model, inputs = [dropdown,devicedropdown])
    devicedropdown.change(fn = load_model, inputs = [dropdown, devicedropdown])
    uploadbtn.upload(fn = upload_save, inputs = uploadbtn, outputs=dropdown)
    deletebtn.click(fn = delete_callback, inputs=dropdown, outputs=dropdown)

    with torch.no_grad():
        image_button.click(yolo_image,image_input,image_output)
        video_button.click(yolo_video,video_input,video_output)
    
    Knowgyu.load(fn = refresh,outputs=dropdown)




if __name__ == "__main__":
    Knowgyu.queue(max_size=10,concurrency_count=2)
    Knowgyu.launch(server_name="0.0.0.0", server_port=9999, show_error=True, inbrowser=True)

    