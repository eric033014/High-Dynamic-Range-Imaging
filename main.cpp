#include "src/hdr.h"
#include "iostream"
using namespace std;

int main(int argc, char **argv) {

    cout << "Starting" << endl;

    HDR main_process;
    main_process.getExposureTime(argv[1]);
    main_process.getInputImage(argv[1]);
    main_process.MTB();
    main_process.getRadianceMap();
    main_process.writeHdrImage();
    main_process.toneMapping();
    main_process.writeJpgImage();

    cout << "Finished" << endl;
    return 0;

}
