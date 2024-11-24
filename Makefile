NVCC = nvcc

CFLAGS = -I/usr/include/opencv4
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgcodecs -lopencv_highgui

TARGET = canny_edge_detection
SRC = canny_edge_detection.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) input.jpg output.jpg
