from ultralytics import YOLO

def compile_models():
    print("Loading PyTorch model (yolo11n.pt)...")
    # Load the base PyTorch model
    model = YOLO('yolo11n.pt')

    print("\n[1/2] Compiling to ONNX format...")
    print("ONNX is highly standardized and provides excellent CPU speedups natively.")
    # Exporting to ONNX
    model.export(format='onnx', imgsz=640)

    print("\n[2/2] Compiling to OpenVINO format...")
    print("OpenVINO is specifically optimized for Intel/AMD CPUs to squeeze out maximum FPS.")
    # Exporting to OpenVINO
    model.export(format='openvino', imgsz=640)

    print("\n✅ Compilation complete!")
    print("To use the compiled models, update sahi_pipeline.py:")
    print("Change: model_path='yolo11n.pt'")
    print("To:     model_path='yolo11n.onnx' or model_path='yolo11n_openvino_model/'")

if __name__ == "__main__":
    compile_models()
