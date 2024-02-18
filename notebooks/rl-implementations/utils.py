import psutil
import ctypes
import json


def get_process_id_by_name(process_name):
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == process_name:
            return process.info['pid']
    return None


def get_active_window_process_id():
    active_window = ctypes.windll.user32.GetForegroundWindow()
    process_id = ctypes.c_ulong(0)
    ctypes.windll.user32.GetWindowThreadProcessId(active_window, ctypes.byref(process_id))
    return process_id.value


def check_if_tm2020_active():
    process_name = "Trackmania.exe"  # Назва процесу, для якого ви хочете отримати ідентифікатор
    process_id = get_process_id_by_name(process_name)

    active_process_id = get_active_window_process_id()  

    if process_id == active_process_id:
        return True
    
    return False


def load_json_file(json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        return data
