// functions for getting data from a PanoSETI HDF5 file

#include "ph5.h"

int PH5::open(const char* path) {
    file_id =  H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (!file_id) return -1;
    return 0;
}

void PH5::close() {
    H5Fclose(file_id);
}

int PH5::get_attr(const char* name, string& value) {
    hid_t att_id = H5Aopen(file_id, name, H5P_DEFAULT);
    if (!att_id) return -1;
    hsize_t sz = H5Aget_storage_size(att_id);
    char buf[sz+1];
    hid_t atype = H5Aget_type(att_id);
    H5Aread(att_id, atype, &buf);
    value = buf;
    H5Aclose(att_id);
    H5Tclose(atype);
    return 0;
}

int PH5::get_frame_set(const char* base_name, int ifs, FRAME_SET& fs) {
    char name[256];
    sprintf(name, "%s%09d", base_name, ifs);
    printf("looking up dataset %s\n", name);
    hid_t dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);
    if (!dataset_id) {
        printf("no dataset\n");
        return -1;
    }
    hsize_t dataset_size = H5Dget_storage_size(dataset_id);
    fs.data = (uint16_t *) malloc(dataset_size);
    //printf("ifg %d size: %d malloced %lx\n", ifs, dataset_size, fs.data);
    herr_t status = H5Dread(
        dataset_id,
        H5Dget_type(dataset_id), H5S_ALL, H5S_ALL, H5P_DEFAULT, fs.data
    );
    if (status) {
        printf("status: %d\n", status);
        return -1;
    }
    fs.nframe_pairs = dataset_size/(FRAME_PAIR_PIXELS*sizeof(uint16_t));
    return 0;
}
