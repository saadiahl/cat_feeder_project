'''
contributors: saadiahlazim4@gmail.com
'''

import numpy as np
import torch, cv2, timm, time
import threading
import coremltools as ct
import PIL.Image
import serial
import time
from support_update import update
from torchvision.transforms import v2, InterpolationMode
from cat import Cat
from statistics import mean, mode
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)
# executorGetWeight = ThreadPoolExecutor(max_workers=2)

# Establish a serial connection
ser = serial.Serial('/dev/cu.usbmodem1201', 115200, timeout=.1) # Replace with the correct port
time.sleep(2)  # wait for the serial connection to initialize

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
print(device)

z_proto = torch.load("./proto.pt")
class_dict = torch.load("./class_dict.pt")
cats = torch.load("./cats.pt")

def open():
    ser.write(b'open\n')  # Sending the open command to Arduino
    # time.sleep(0.8)

def close():
    ser.write(b'close\n')  # Sending the close command to Arduino
    # time.sleep(0.8)

def dispenseFood(portion: float) -> float:
    command = f"dispense {portion}\n".encode()  # Encode the command to bytes
    ser.write(command)
    # time.sleep(0.8)
    time.sleep(5)
    return getCurrentFoodWeight()

def getCurrentFoodWeight() -> float:
    ser.write(b'get_weight\n')
    # time.sleep(0.3)  # Must be higher than arduino loop delay
    while True:
        if ser.in_waiting:
            # time.sleep(0.3)
            try:
                weight = float(ser.readline().decode().strip())
                return weight
                # if weight != 0.0 and weight > 0.0:
                #     return weight
                # else:
                #     ser.write(b'get_weight\n')
                #     time.sleep(0.3)
            except ValueError:
                print("Received non-float data")
                ser.write(b'get_weight\n')
                # time.sleep(0.3)
                continue

def fsl(image: np.ndarray) -> str:
    image_size = 336
    normalize = v2.Normalize(mean = [0.48145466, 0.4578275, 0.40821073],
                                    std = [0.26862954, 0.26130258, 0.27577711])

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True), 
        v2.Resize([image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.RandomEqualize(1.0),
        v2.ToDtype(torch.float32, scale=True),
        normalize
    ])

    query_image = transforms(image)
    with torch.no_grad():
        z_query = modelFSL.forward(query_image.to(device, non_blocking=True).unsqueeze(0))        
    dists = torch.cdist(z_query, z_proto)
    scores = -dists #? redundant
    _, predicted_labels = torch.max(scores.data, 1)
    return class_dict[int(predicted_labels)]

def getCatNear(image: np.ndarray, confidenceThreshold: float=0.7, nearThreshold: float=0.40) -> bool:
    pil_img = PIL.Image.fromarray(image)
    resizedImg = pil_img.resize([640,640])
    out_dict = modelObjDet.predict({'image': resizedImg})
    try:
        index_max = max(range(80), key=out_dict["confidence"][0].__getitem__)
    except:
        return False

    if index_max == 15:
        if out_dict["confidence"][0][15] > confidenceThreshold:
            if mean(out_dict["coordinates"][0][2:]) > nearThreshold:
                return True
    return False


def main(event: threading.Event=False) -> None:
    cap = cv2.VideoCapture(0)

    # imageSize = 640
    # cap.set(3, imageSize)
    # cap.set(4, imageSize)
    cap.set(3, 1552)
    cap.set(4, 1552)
    cap.set(5, 60)

    if event:
        event.wait()

    mainEating: bool = False
    qCatNameLookBackWindow: int = 40
    qCatName: list[str] = [] #* requires iterable datatype
    
    qCatNearLookBackWindow: int = 40 #? must be higher than qCatNameLookBackWindow
    qCatNear: list[bool] = [False] * qCatNearLookBackWindow #* requires iterable datatype

    # Initialize the time and counter variables
    start_time = time.time()
    num_iterations = 0
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        
        cv2.imshow('Webcam', img)
        # cv2.imshow('Webcam', cv2.flip(img, 1))

        #! STATE MANAGEMENT
        '''
        #* Cases
        1. One-by-one eating. Current
        2. Eating interfering, Requires temporal cat_name averages and interfering tolerance.

        init where

        keep cats in dict for key: value retrival?

        #* One True at most
        entering = True
        eating = False

        #* To calculate session eating weight current-initial and update
        initialFoodWeight = 0
        currentFoodWeight = 0

        #* Normalize for easier interpretation
        distanceTolerance = float
        interferingTolerance = float

        #! implement cat object detection (cat enters event)
        if cat:
            distance = check bounding box width/height with frame width/height
            if distance == close +- distanceTolerance:
                cat_name = fsl_event()
                allow: bool = getSession()
                if allow:
                    #! arduino event, open()
                    portion = getSessionPortion(), given cat_name
                    if entering:
                        #! arduino event, dispenseFood(portion)
                        initialFoodWeight = portion
                        entering = False
                        eating = True
                    else:
                        currentFoodWeight = #! arduino event, getCurrentFoodWeight()
                        eating = True #? redundant
            elif distance == far +- distanceTolerance && eating == True:
                #! arduino event, close()
                eating = False
                entering = True
                setSessionPortion(initialFoodWeight-currentFoodWeight), given cat_name
        '''

        catNear = getCatNear(img)
        # print("catNear", catNear, qCatNear)
        qCatNear.append(catNear)
        if catNear and mode(qCatNear) == True: #? and
            catName = fsl(img)
            qCatName.append(catName)
            allow = cats[catName].getSession()
            portion = cats[catName].getSessionPortion()

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(catName, 0, current_time, portion, mode(qCatName))
            if allow and mode(qCatName) == catName:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(catName, 1, current_time, portion, mode(qCatName))
                if cats[catName].entering and mainEating == False:
                    open()
                    # initialFoodWeight = threading.Thread(target=dispenseFood, args=(portion,)).start()
                    futureDispense = executor.submit(dispenseFood, portion)
                    # dispenseFood(portion)
                    # initialFoodWeight = getCurrentFoodWeight()
                    # print(catName, 1, portion)
                    # print(catName, 1, initialFoodWeight)
                    # initialFoodWeight = portion
                    cats[catName].entering = False
                    cats[catName].eating = True
                    mainEating = True
                    currentFoodWeight = 0
                else:
                    if futureDispense.done():
                        initialFoodWeight = futureDispense.result() 
                        # currentFoodWeight = getCurrentFoodWeight()
                        futureCurrentFoodWeight = executor.submit(getCurrentFoodWeight)
                    # currentFoodWeight = 10
            elif mode(qCatName) != catName:
                catName = mode(qCatName)
            print(currentFoodWeight, futureDispense.running(), futureDispense.done())
        elif catNear == False and mainEating == True and cats[catName].eating == True and mode(qCatNear) == False and initialFoodWeight != None: #! multiple 2's
            close()       
            cats[catName].eating = False
            mainEating = False
            cats[catName].entering = True
            # initialFoodWeight = futureDispense.result() 
            try:
                currentFoodWeight = futureCurrentFoodWeight.result()
            except:
                print("Failed to retrieve weight")
                continue 
            cats[catName].setSessionPortion(initialFoodWeight-currentFoodWeight)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(catName, 2, current_time, portion, mode(qCatName)) 
            
        # elif catNear == False:
        #     qCatNear.append(False)

        if len(qCatName) >= qCatNameLookBackWindow: 
            qCatName.pop(0)
        if len(qCatNear) >= qCatNearLookBackWindow: 
            qCatNear.pop(0)

        # str_labels = fsl(img)
        # print(str_labels)

        # Increment the iteration count
        num_iterations += 1

        if cv2.waitKey(1) == ord('q'):
            torch.save(cats.copy(), "./runtimeCats.pt")
            break

    # Calculate the time elapsed and iterations per second
    end_time = time.time()
    time_elapsed = end_time - start_time
    iterations_per_second = num_iterations / time_elapsed

    print(f"Total time elapsed: {time_elapsed} seconds")
    print(f"Iterations per second: {iterations_per_second}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True
    modelFSL = timm.create_model("eva02_tiny_patch14_336.mim_in22k_ft_in1k", pretrained=True, num_classes=0, scriptable=True)
    modelFSL = modelFSL.to(device)
    modelFSL.eval()
    # modelFSL = torch.compile(modelFSL, fullgraph=False)

    modelObjDet = ct.models.CompiledMLModel("yolov8m.mlmodelc", ct.ComputeUnit.ALL)

    print("Update=1, else 0")
    update_bool = 0
    # update_bool = int(input())

    if update_bool == 1:
        # # Create an Event object for synchronization
        # update_completed_event = threading.Event()

        # # Create and start the update thread
        # update_thread = threading.Thread(target=update, args=(update_completed_event, model))
        # update_thread.start()

        # # Run the main function on the main thread
        # main(update_completed_event, model)

        # # Optionally, wait for the update thread to complete
        # update_thread.join()
        update(event=False, model=modelFSL)
        main(event=False)
    else:
        main(event=False)