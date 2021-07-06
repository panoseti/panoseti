/* HSD_output_thread.c
 *
 * Writes the data to HDF5 output file
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include "hashpipe.h"
#include "HSD_databuf.h"
#include "hiredis/hiredis.h"
#include "hdf5.h"
#include "hdf5_hl.h"

//Defining the names of redis keys and files
#define OBSERVATORY "LICK"
#define GPSPRIMNAME "GPSPRIM"
#define GPSSUPPNAME "GPSSUPP"
#define WRSWITCHNAME "WRSWITCH"

//Defining the Formats that will be used within the HDF5 data file
#define H5FILE_NAME_FORMAT "PANOSETI_%s_%04i_%02i_%02i_%02i-%02i-%02i.h5"
#define TIME_FORMAT "%04i-%02i-%02iT%02i:%02i:%02i UTC"
#define FRAME_FORMAT "Frame%05i"
#define IMGDATA_FORMAT "DATA%09i"
#define IMGDATA_META_FORMAT "DATA%09i_%s"
#define PHDATA_FORMAT "PH_Module%05i_Quabo%01i_UTC%09i_NANOSEC%09i_PKTNUM%05i"
#define QUABO_FORMAT "QUABO%05i_%01i"

#define HK_TABLENAME_FORAMT "HK_Module%05i_Quabo%01i"
#define HK_TABLETITLE_FORMAT "HouseKeeping Data for Module%05i_Quabo%01i"

#define IMGHeadFIELDS 4

//Defining the string buffer size
#define STRBUFFSIZE 80

static char saveLocation[STRBUFFSIZE];

//Defining the static values for the storage values for HDF5 file
static hsize_t storageDim[RANK] = {PKTPERDATASET, PKTPERPAIR, SCIDATASIZE};
static hid_t storageSpace = H5Screate_simple(RANK, storageDim, NULL);

static hsize_t storageDimPH[RANK] = {PKTPERDATASET, 1, SCIDATASIZE};
static hid_t storageSpacePH = H5Screate_simple(RANK, storageDimPH, NULL);

static hsize_t storageDimMeta[RANK] = {PKTPERDATASET, PKTPERPAIR, 1};
static hid_t storageSpaceMeta = H5Screate_simple(RANK, storageDimMeta, NULL);

static hsize_t storageDimModPair[RANK] = {PKTPERDATASET, 1, 1};
static hid_t storageSpaceModPair = H5Screate_simple(RANK, storageDimModPair, NULL);

static hsize_t chunkDim[RANK] = {1, PKTPERPAIR, SCIDATASIZE};
static hid_t creation_property = H5Pcreate(H5P_DATASET_CREATE);

static hid_t storageTypebit16 = H5Tcopy(H5T_STD_U16LE);
static hid_t storageTypebit8 = H5Tcopy(H5T_STD_U8LE);

static long long fileSize = 0;

static long long maxFileSize = 0; //IN UNITS OF APPROX 2 BYTES OR 16 bits

/**
 * The fileID structure for the current HDF5 opened.
 */
typedef struct fileIDs
{
    hid_t file; /* file and dataset handles */
    hid_t bit16IMGData, bit8IMGData, PHData, ShortTransient, bit16HCData, bit8HCData, DynamicMeta, StaticMeta;
} fileIDs_t;

/**
 * Module Pair structure to store data information regarding storing in HDF5
 */
typedef struct modulePairFile
{
    unsigned int mod1Name;
    unsigned int mod2Name;

    hid_t bit16IMGGroup;
    hid_t bit16Dataset;
    hid_t bit16pktNum;
    hid_t bit16pktNSEC;
    hid_t bit16tv_sec;
    hid_t bit16tv_usec;
    hid_t bit16status;
    uint32_t bit16DatasetIndex;
    uint32_t bit16ModPairIndex;

    hid_t bit8IMGGroup;
    hid_t bit8Dataset;
    hid_t bit8pktNum;
    hid_t bit8pktNSEC;
    hid_t bit8tv_sec;
    hid_t bit8tv_usec;
    hid_t bit8status;
    uint32_t bit8DatasetIndex;
    uint32_t bit8ModPairIndex;

    hid_t PHGroup;
    hid_t PHDataset;
    hid_t PHpktNum;
    hid_t PHpktNSEC;
    hid_t PHtv_sec;
    hid_t PHtv_usec;
    hid_t PHpktUTC;
    hid_t PHmodNum;
    hid_t PHquaNum;
    uint32_t PHDatasetIndex;
    uint32_t PHModPairIndex;

    modulePairFile *next_modulePairFile;
} modulePairFile_t;
/**
 * Instantiate the dataset for a modulepair 
 * @param modPair The module pair object for which we are creating
 * @param acqmode The acquisition mode that we have (possible values are 16 and 8)
 */
void create_ModPair_Dataset(modulePairFile_t *modPair, int acqmode) {
    char name[STRBUFFSIZE];
    if (acqmode == 16) {

        if (modPair->bit16Dataset >= 0) {
            H5Dclose(modPair->bit16Dataset);
            H5Dclose(modPair->bit16pktNum);
            H5Dclose(modPair->bit16pktNSEC);
            H5Dclose(modPair->bit16tv_sec);
            H5Dclose(modPair->bit16tv_usec);
            H5Dclose(modPair->bit16status);
        }
        modPair->bit16ModPairIndex = 0;
        modPair->bit16DatasetIndex += 1;

        sprintf(name, IMGDATA_FORMAT, modPair->bit16DatasetIndex);
        modPair->bit16Dataset = H5Dcreate2(modPair->bit16IMGGroup, name, storageTypebit16, storageSpace, H5P_DEFAULT, creation_property, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit16DatasetIndex, "pktNum");
        modPair->bit16pktNum = H5Dcreate2(modPair->bit16IMGGroup, name, H5T_STD_U16LE, storageSpaceMeta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit16DatasetIndex, "pktNSEC");
        modPair->bit16pktNSEC = H5Dcreate2(modPair->bit16IMGGroup, name, H5T_STD_U32LE, storageSpaceMeta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit16DatasetIndex, "tv_sec");
        modPair->bit16tv_sec = H5Dcreate2(modPair->bit16IMGGroup, name, H5T_NATIVE_LONG, storageSpaceMeta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit16DatasetIndex, "tv_usec");
        modPair->bit16tv_usec = H5Dcreate2(modPair->bit16IMGGroup, name, H5T_NATIVE_LONG, storageSpaceMeta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit16DatasetIndex, "status");
        modPair->bit16status = H5Dcreate2(modPair->bit16IMGGroup, name, H5T_NATIVE_LONG, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    } else if (acqmode == 8){

        if (modPair->bit8Dataset >= 0) {
            H5Dclose(modPair->bit8Dataset);
            H5Dclose(modPair->bit8pktNum);
            H5Dclose(modPair->bit8pktNSEC);
            H5Dclose(modPair->bit8tv_sec);
            H5Dclose(modPair->bit8tv_usec);
        }
        modPair->bit8ModPairIndex = 0;
        modPair->bit8DatasetIndex += 1;

        sprintf(name, IMGDATA_FORMAT, modPair->bit8DatasetIndex);
        modPair->bit8Dataset = H5Dcreate2(modPair->bit8IMGGroup, name, storageTypebit8, storageSpace, H5P_DEFAULT, creation_property, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit8DatasetIndex, "pktNum");
        modPair->bit8pktNum = H5Dcreate2(modPair->bit8IMGGroup, name, H5T_STD_U16LE, storageSpaceMeta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit8DatasetIndex, "pktNSEC");
        modPair->bit8pktNSEC = H5Dcreate2(modPair->bit8IMGGroup, name, H5T_STD_U32LE, storageSpaceMeta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit8DatasetIndex, "tv_sec");
        modPair->bit8tv_sec = H5Dcreate2(modPair->bit8IMGGroup, name, H5T_NATIVE_LONG, storageSpaceMeta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit8DatasetIndex, "tv_usec");
        modPair->bit8tv_usec = H5Dcreate2(modPair->bit8IMGGroup, name, H5T_NATIVE_LONG, storageSpaceMeta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->bit8DatasetIndex, "status");
        modPair->bit8status = H5Dcreate2(modPair->bit8IMGGroup, name, H5T_NATIVE_LONG, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    } else {

        if (modPair->PHDataset >= 0) {
            H5Dclose(modPair->PHDataset);
            H5Dclose(modPair->PHpktNum);
            H5Dclose(modPair->PHpktNSEC);
            H5Dclose(modPair->PHpktUTC);
            H5Dclose(modPair->PHtv_sec);
            H5Dclose(modPair->PHtv_usec);
            H5Dclose(modPair->PHmodNum);
            H5Dclose(modPair->PHquaNum);
        }
        modPair->PHModPairIndex = 0;
        modPair->PHDatasetIndex += 1;

        sprintf(name, IMGDATA_FORMAT, modPair->PHDatasetIndex);
        modPair->PHDataset = H5Dcreate2(modPair->PHGroup, name, storageTypebit16, storageSpacePH, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->PHDatasetIndex, "modNum");
        modPair->PHmodNum = H5Dcreate2(modPair->PHGroup, name, H5T_STD_U16LE, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->PHDatasetIndex, "quaNum");
        modPair->PHquaNum = H5Dcreate2(modPair->PHGroup, name, H5T_STD_U8LE, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->PHDatasetIndex, "pktNum");
        modPair->PHpktNum = H5Dcreate2(modPair->PHGroup, name, H5T_STD_U16LE, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->PHDatasetIndex, "pktNSEC");
        modPair->PHpktNSEC = H5Dcreate2(modPair->PHGroup, name, H5T_STD_U32LE, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->PHDatasetIndex, "pktUTC");
        modPair->PHpktUTC = H5Dcreate2(modPair->PHGroup, name, H5T_STD_U32LE, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->PHDatasetIndex, "tv_sec");
        modPair->PHtv_sec = H5Dcreate2(modPair->PHGroup, name, H5T_NATIVE_LONG, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        sprintf(name, IMGDATA_META_FORMAT, modPair->PHDatasetIndex, "tv_usec");
        modPair->PHtv_usec = H5Dcreate2(modPair->PHGroup, name, H5T_NATIVE_LONG, storageSpaceModPair, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
}

/**
 * Initializing an empty modulePairFile object
 */
modulePairFile_t *modulePairFile_t_new(fileIDs_t *currFile, uint16_t mod1, uint16_t mod2, int createGroup = 1) {

    modulePairFile_t *newModPair = (modulePairFile_t *)malloc(sizeof(struct modulePairFile));
    if (newModPair == NULL) {
        printf("Error: Unable to malloc space for ModulePairFile\n");
        exit(1);
    }

    char name[STRBUFFSIZE];

    newModPair->mod1Name = mod1;
    newModPair->mod2Name = mod2;
    if (createGroup) {
        sprintf(name, MODULEPAIR_FORMAT, mod1, mod2);
        newModPair->bit16IMGGroup = H5Gcreate(currFile->bit16IMGData, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        newModPair->bit8IMGGroup = H5Gcreate(currFile->bit8IMGData, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        newModPair->PHGroup = H5Gcreate(currFile->PHData, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    newModPair->bit16Dataset = -1;
    newModPair->bit16pktNum = -1;
    newModPair->bit16pktNSEC = -1;
    newModPair->bit16tv_sec = -1;
    newModPair->bit16tv_usec = -1;
    newModPair->bit16status = -1;
    newModPair->bit16DatasetIndex = -1;
    newModPair->bit16ModPairIndex = PKTPERDATASET;

    newModPair->bit8Dataset = -1;
    newModPair->bit8pktNum = -1;
    newModPair->bit8pktNSEC = -1;
    newModPair->bit8tv_sec = -1;
    newModPair->bit8tv_usec = -1;
    newModPair->bit8status = -1;
    newModPair->bit8DatasetIndex = -1;
    newModPair->bit8ModPairIndex = PKTPERDATASET;

    newModPair->PHDataset = -1;
    newModPair->PHmodNum = -1;
    newModPair->PHquaNum = -1;
    newModPair->PHpktNum = -1;
    newModPair->PHpktNSEC = -1;
    newModPair->PHpktUTC = -1;
    newModPair->PHtv_sec = -1;
    newModPair->PHtv_usec = -1;
    newModPair->PHDatasetIndex = -1;
    newModPair->PHModPairIndex = PKTPERDATASET;

    newModPair->next_modulePairFile = NULL;
    return newModPair;
}

/**
 * Writes the IMG dataset into the file given
 * @param modPair The module pair object associated with the pair of modules specificed in config file
 * @param block The datablock from the output buffer that contains the module pair data
 * @param i The index of the block that we are looking at
 */
void write_Dataset(modulePairFile_t *modPair, HSD_output_block_t *block, int i) {
    hid_t status;
    uint32_t modulePairIndex;
    int mode = block->header.acqmode[i];
    if (mode == 16) {
        modulePairIndex = modPair->bit16ModPairIndex;
    } else if (mode == 8) {
        modulePairIndex = modPair->bit8ModPairIndex;
    }

    hsize_t offset[RANK] = {modulePairIndex, 0, 0};
    hsize_t count[RANK] = {1, PKTPERPAIR, SCIDATASIZE};

    hid_t dataSpace = H5Screate_simple(RANK, storageDim, NULL);
    //Create the dataspace within the HDF5 dataset that will be written
    H5Sselect_hyperslab(dataSpace, H5S_SELECT_SET, offset, NULL, count, NULL);

    hsize_t mOffset[RANK - 1] = {0, 0};
    hsize_t mCount[RANK - 1] = {PKTPERPAIR, SCIDATASIZE};

    hid_t dataMSpace = H5Screate_simple(RANK - 1, mCount, NULL);
    //Create the dataspace of the data within memory
    H5Sselect_hyperslab(dataMSpace, H5S_SELECT_SET, mOffset, NULL, mCount, NULL);


    hsize_t offsetMeta[RANK] = {modulePairIndex, 0, 0};
    hsize_t countMeta[RANK] = {1, PKTPERPAIR, 1};

    hid_t dataSpaceMeta = H5Screate_simple(RANK, storageDimMeta, NULL);
    //Create the dataspace within the HDF5 dataset for metadata
    H5Sselect_hyperslab(dataSpaceMeta, H5S_SELECT_SET, offsetMeta, NULL, countMeta, NULL);

    hsize_t mOffsetMeta[RANK - 1] = {0, 0};
    hsize_t mCountMeta[RANK - 1] = {PKTPERPAIR, 1};

    hid_t dataMSpaceMeta = H5Screate_simple(RANK - 1, mCountMeta, NULL);
    //Create the dataspace of the data within memory
    H5Sselect_hyperslab(dataMSpaceMeta, H5S_SELECT_SET, mOffsetMeta, NULL, mCountMeta, NULL);


    hsize_t offsetModPair[RANK] = {modulePairIndex, 0, 0};
    hsize_t countModPair[RANK] = {1, 1, 1};

    hid_t dataSpaceModPair = H5Screate_simple(RANK, storageDimModPair, NULL);
    //Create the dataspace within the HDF5 dataset for metadata
    H5Sselect_hyperslab(dataSpaceModPair, H5S_SELECT_SET, offsetModPair, NULL, countModPair, NULL);

    hsize_t mOffsetModPair[RANK - 1] = {0, 0};
    hsize_t mCountModPair[RANK - 1] = {1, 1};

    hid_t dataMSpaceModPair = H5Screate_simple(RANK - 1, mCountModPair, NULL);
    //Create the dataspace of the data within memory
    H5Sselect_hyperslab(dataMSpaceModPair, H5S_SELECT_SET, mOffsetModPair, NULL, mCountModPair, NULL);

    if (mode == 16) {
        status = H5Dwrite(modPair->bit16Dataset, storageTypebit16, dataMSpace, dataSpace, H5P_DEFAULT, block->stream_block + (i * MODPAIRDATASIZE));
        status = H5Dwrite(modPair->bit16pktNum, H5T_STD_U16LE, dataMSpaceMeta, dataSpaceMeta, H5P_DEFAULT, block->header.pktNum + (i * PKTPERPAIR));
        status = H5Dwrite(modPair->bit16pktNSEC, H5T_STD_U32LE, dataMSpaceMeta, dataSpaceMeta, H5P_DEFAULT, block->header.pktNSEC + (i * PKTPERPAIR));
        status = H5Dwrite(modPair->bit16tv_sec, H5T_NATIVE_LONG, dataMSpaceMeta, dataSpaceMeta, H5P_DEFAULT, block->header.tv_sec + (i * PKTPERPAIR));
        status = H5Dwrite(modPair->bit16tv_usec, H5T_NATIVE_LONG, dataMSpaceMeta, dataSpaceMeta, H5P_DEFAULT, block->header.tv_usec + (i * PKTPERPAIR));
        status = H5Dwrite(modPair->bit16status, H5T_STD_U8LE, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.status + i);
    } else if (mode == 8) {
        status = H5Dwrite(modPair->bit8Dataset, storageTypebit8, dataMSpace, dataSpace, H5P_DEFAULT, block->stream_block + (i * MODPAIRDATASIZE));
        status = H5Dwrite(modPair->bit8pktNum, H5T_STD_U16LE, dataMSpaceMeta, dataSpaceMeta, H5P_DEFAULT, block->header.pktNum + (i * PKTPERPAIR));
        status = H5Dwrite(modPair->bit8pktNSEC, H5T_STD_U32LE, dataMSpaceMeta, dataSpaceMeta, H5P_DEFAULT, block->header.pktNSEC + (i * PKTPERPAIR));
        status = H5Dwrite(modPair->bit8tv_sec, H5T_NATIVE_LONG, dataMSpaceMeta, dataSpaceMeta, H5P_DEFAULT, block->header.tv_sec + (i * PKTPERPAIR));
        status = H5Dwrite(modPair->bit8tv_usec, H5T_NATIVE_LONG, dataMSpaceMeta, dataSpaceMeta, H5P_DEFAULT, block->header.tv_usec + (i * PKTPERPAIR));
        status = H5Dwrite(modPair->bit8status, H5T_STD_U8LE, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.status + i);
    }
    #ifdef TEST_MODE
        printf("Acqmode: %u, modPairIndex: %u\n", mode, modulePairIndex);
    #endif

    H5Sclose(dataSpace);
    H5Sclose(dataMSpace);
    H5Sclose(dataSpaceMeta);
    H5Sclose(dataMSpaceMeta);
    H5Sclose(dataSpaceModPair);
    H5Sclose(dataMSpaceModPair);
}

/**
 * Writes the PH dataset into the file given
 * @param modPair The module pair object associated with the pair of modules specificed in config file
 * @param block The datablock from the output buffer that contains the module pair data
 * @param i The index of the block that we are looking at
 */
void write_PHDataset(modulePairFile_t *modPair, HSD_output_block_t *block, int i) {
    hid_t status;

    hsize_t offsetPH[RANK] = {modPair->PHModPairIndex, 0, 0};
    hsize_t countPH[RANK] = {1, 1, SCIDATASIZE};

    hid_t dataSpacePH = H5Screate_simple(RANK, storageDimPH, NULL);
    //Create the dataspace within the HDF5 dataset for metadata
    H5Sselect_hyperslab(dataSpacePH, H5S_SELECT_SET, offsetPH, NULL, countPH, NULL);


    hsize_t mOffsetPH[RANK-1] = {0, 0};
    hsize_t mCountPH[RANK-1] = {1, SCIDATASIZE};

    hid_t dataMSpacePH = H5Screate_simple(RANK-1, mCountPH, NULL);
    //Create the dataspace of the data within memory
    H5Sselect_hyperslab(dataMSpacePH, H5S_SELECT_SET, mOffsetPH, NULL, mCountPH, NULL);



    hsize_t offsetModPair[RANK] = {modPair->PHModPairIndex, 0, 0};
    hsize_t countModPair[RANK] = {1, 1, 1};

    hid_t dataSpaceModPair = H5Screate_simple(RANK, storageDimModPair, NULL);
    //Create the dataspace within the HDF5 dataset for metadata
    H5Sselect_hyperslab(dataSpaceModPair, H5S_SELECT_SET, offsetModPair, NULL, countModPair, NULL);

    hsize_t mOffsetModPair[RANK - 1] = {0, 0};
    hsize_t mCountModPair[RANK - 1] = {1, 1};

    hid_t dataMSpaceModPair = H5Screate_simple(RANK - 1, mCountModPair, NULL);
    //Create the dataspace of the data within memory
    H5Sselect_hyperslab(dataMSpaceModPair, H5S_SELECT_SET, mOffsetModPair, NULL, mCountModPair, NULL);

    status = H5Dwrite(modPair->PHDataset, storageTypebit16, dataMSpacePH, dataSpacePH, H5P_DEFAULT, block->coinc_block + (i * PKTDATASIZE));
    status = H5Dwrite(modPair->PHmodNum, H5T_STD_U16LE, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.coin_modNum + i);
    status = H5Dwrite(modPair->PHquaNum, H5T_STD_U8LE, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.coin_quaNum + i);
    status = H5Dwrite(modPair->PHpktNum, H5T_STD_U16LE, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.coin_pktNum + i);
    status = H5Dwrite(modPair->PHpktNSEC, H5T_STD_U32LE, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.coin_pktNSEC + i);
    status = H5Dwrite(modPair->PHpktUTC, H5T_STD_U32LE, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.coin_pktUTC + i);
    status = H5Dwrite(modPair->PHtv_sec, H5T_NATIVE_LONG, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.coin_tv_sec + i);
    status = H5Dwrite(modPair->PHtv_usec, H5T_NATIVE_LONG, dataMSpaceModPair, dataSpaceModPair, H5P_DEFAULT, block->header.coin_tv_usec + i);

    H5Sclose(dataSpacePH);
    H5Sclose(dataMSpacePH);
    H5Sclose(dataSpaceModPair);
    H5Sclose(dataMSpaceModPair);
}

typedef struct HKPackets {
    char SYSTIME[STRBUFFSIZE];
    uint16_t BOARDLOC;
    float HVMON0, HVMON1, HVMON2, HVMON3;
    float HVIMON0, HVIMON1, HVIMON2, HVIMON3;
    float RAWHVMON;
    float V12MON, V18MON, V33MON, V37MON;
    float I10MON, I18MON, I33MON;
    float TEMP1;
    float TEMP2;
    float VCCINT, VCCAUX;
    uint64_t UID;
    uint8_t SHUTTER_STATUS, LIGHT_STATUS;
    uint32_t FWID0, FWID1;
} HKPackets_t;

const HKPackets_t HK_dst_buf[0] = {};

const size_t HK_dst_size = sizeof(HKPackets_t);

const size_t HK_dst_offset[HKFIELDS] = {HOFFSET(HKPackets_t, SYSTIME),
                                        HOFFSET(HKPackets_t, BOARDLOC),
                                        HOFFSET(HKPackets_t, HVMON0),
                                        HOFFSET(HKPackets_t, HVMON1),
                                        HOFFSET(HKPackets_t, HVMON2),
                                        HOFFSET(HKPackets_t, HVMON3),
                                        HOFFSET(HKPackets_t, HVIMON0),
                                        HOFFSET(HKPackets_t, HVIMON1),
                                        HOFFSET(HKPackets_t, HVIMON2),
                                        HOFFSET(HKPackets_t, HVIMON3),
                                        HOFFSET(HKPackets_t, RAWHVMON),
                                        HOFFSET(HKPackets_t, V12MON),
                                        HOFFSET(HKPackets_t, V18MON),
                                        HOFFSET(HKPackets_t, V33MON),
                                        HOFFSET(HKPackets_t, V37MON),
                                        HOFFSET(HKPackets_t, I10MON),
                                        HOFFSET(HKPackets_t, I18MON),
                                        HOFFSET(HKPackets_t, I33MON),
                                        HOFFSET(HKPackets_t, TEMP1),
                                        HOFFSET(HKPackets_t, TEMP2),
                                        HOFFSET(HKPackets_t, VCCINT),
                                        HOFFSET(HKPackets_t, VCCAUX),
                                        HOFFSET(HKPackets_t, UID),
                                        HOFFSET(HKPackets_t, SHUTTER_STATUS),
                                        HOFFSET(HKPackets_t, LIGHT_STATUS),
                                        HOFFSET(HKPackets_t, FWID0),
                                        HOFFSET(HKPackets_t, FWID1)};

const size_t HK_dst_sizes[HKFIELDS] = {sizeof(HK_dst_buf[0].SYSTIME),
                                       sizeof(HK_dst_buf[0].BOARDLOC),
                                       sizeof(HK_dst_buf[0].HVMON0),
                                       sizeof(HK_dst_buf[0].HVMON1),
                                       sizeof(HK_dst_buf[0].HVMON2),
                                       sizeof(HK_dst_buf[0].HVMON3),
                                       sizeof(HK_dst_buf[0].HVIMON0),
                                       sizeof(HK_dst_buf[0].HVIMON1),
                                       sizeof(HK_dst_buf[0].HVIMON2),
                                       sizeof(HK_dst_buf[0].HVIMON3),
                                       sizeof(HK_dst_buf[0].RAWHVMON),
                                       sizeof(HK_dst_buf[0].V12MON),
                                       sizeof(HK_dst_buf[0].V18MON),
                                       sizeof(HK_dst_buf[0].V33MON),
                                       sizeof(HK_dst_buf[0].V37MON),
                                       sizeof(HK_dst_buf[0].I10MON),
                                       sizeof(HK_dst_buf[0].I18MON),
                                       sizeof(HK_dst_buf[0].I33MON),
                                       sizeof(HK_dst_buf[0].TEMP1),
                                       sizeof(HK_dst_buf[0].TEMP2),
                                       sizeof(HK_dst_buf[0].VCCINT),
                                       sizeof(HK_dst_buf[0].VCCAUX),
                                       sizeof(HK_dst_buf[0].UID),
                                       sizeof(HK_dst_buf[0].SHUTTER_STATUS),
                                       sizeof(HK_dst_buf[0].LIGHT_STATUS),
                                       sizeof(HK_dst_buf[0].FWID0),
                                       sizeof(HK_dst_buf[0].FWID1)};

const char *HK_field_names[HKFIELDS] = {"SYSTIME", "BOARDLOC",
                                        "HVMON0", "HVMON1", "HVMON2", "HVMON3",
                                        "HVIMON0", "HVIMON1", "HVIMON2", "HVIMON3",
                                        "RAWHVMON",
                                        "V12MON", "V18MON", "V33MON", "V37MON",
                                        "I10MON", "I18MON", "I33MON",
                                        "TEMP1", "TEMP2",
                                        "VCCINT", "VCCAUX",
                                        "UID",
                                        "SHUTTER_STATUS", "LIGHT_SENSOR_STATUS",
                                        "FWID0", "FWID1"};

hid_t get_H5T_string_type() {
    hid_t string_type;

    string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, STRBUFFSIZE);
    return string_type;
}

const hid_t HK_field_types[HKFIELDS] = {
    get_H5T_string_type(), H5T_STD_U16LE,                                   // SYSTIME, BOARDLOC
    H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, // HVMON0-3
    H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, // HVIMON0-3
    H5T_NATIVE_FLOAT,                                                       // RAWHVMON
    H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, // V12MON, V18MON, V33MON, V37MON
    H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,                   // I10MON, I18MON, I33MON
    H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,                                     // TEMP1, TEMP2
    H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,                                     // VCCINT, VCCAUX
    H5T_STD_U64LE,                                                          // UID
    H5T_STD_I8LE, H5T_STD_I8LE,                                             // SHUTTER and LIGHT_SENSOR STATUS
    H5T_STD_U32LE, H5T_STD_U32LE                                            // FWID0 and FWID1
};

typedef struct GPSPackets {
    char GPSTIME[STRBUFFSIZE];
    uint32_t TOW;
    uint16_t WEEKNUMBER;
    uint8_t UTCOFFSET;
    char TIMEFLAG[STRBUFFSIZE];
    char PPSFLAG[STRBUFFSIZE];
    uint8_t TIMESET;
    uint8_t UTCINFO;
    uint8_t TIMEFROMGPS;
    char TV_UTC[STRBUFFSIZE];
} GPSPackets_t;

const GPSPackets_t GPS_dst_buf[0] = {};

const size_t GPS_dst_size = sizeof(GPSPackets_t);

const size_t GPS_dst_offset[GPSFIELDS] = {HOFFSET(GPSPackets_t, GPSTIME),
                                          HOFFSET(GPSPackets_t, TOW),
                                          HOFFSET(GPSPackets_t, WEEKNUMBER),
                                          HOFFSET(GPSPackets_t, UTCOFFSET),
                                          HOFFSET(GPSPackets_t, TIMEFLAG),
                                          HOFFSET(GPSPackets_t, PPSFLAG),
                                          HOFFSET(GPSPackets_t, TIMESET),
                                          HOFFSET(GPSPackets_t, UTCINFO),
                                          HOFFSET(GPSPackets_t, TIMEFROMGPS),
                                          HOFFSET(GPSPackets_t, TV_UTC)};

const size_t GPS_dst_sizes[GPSFIELDS] = {sizeof(GPS_dst_buf[0].GPSTIME),
                                         sizeof(GPS_dst_buf[0].TOW),
                                         sizeof(GPS_dst_buf[0].WEEKNUMBER),
                                         sizeof(GPS_dst_buf[0].UTCOFFSET),
                                         sizeof(GPS_dst_buf[0].TIMEFLAG),
                                         sizeof(GPS_dst_buf[0].PPSFLAG),
                                         sizeof(GPS_dst_buf[0].TIMESET),
                                         sizeof(GPS_dst_buf[0].UTCINFO),
                                         sizeof(GPS_dst_buf[0].TIMEFROMGPS),
                                         sizeof(GPS_dst_buf[0].TV_UTC)};

const char *GPS_field_names[GPSFIELDS] = {"GPSTIME",
                                          "TOW",
                                          "WEEKNUMBER",
                                          "UTCOFFSET",
                                          "TIMEFLAG",
                                          "PPSFLAG",
                                          "TIMESET",
                                          "UTCINFO",
                                          "TIMEFROMGPS",
                                          "TV_UTC"};

const hid_t GPS_field_types[GPSFIELDS] = {
    get_H5T_string_type(), // GPSTIME
    H5T_STD_U32LE,         // TOW;
    H5T_STD_U16LE,         // WEEKNUMBER
    H5T_STD_U8LE,          // UTCOFFSET
    get_H5T_string_type(), // TIMEFLAG[STRBUFFSIZE]
    get_H5T_string_type(), // PPSFLAG[STRBUFFSIZE]
    H5T_STD_U8LE,          // TIMESET
    H5T_STD_U8LE,          // UTCINFO
    H5T_STD_U8LE,           // TIMEFROMGPS
    get_H5T_string_type()  // TV_UTC
};

/**
 * Create a singular string attribute attached to the given group.
 */
void createStrAttribute(hid_t group, const char *name, char *data) {
    hid_t datatype, dataspace; /* handles */
    hid_t attribute;

    dataspace = H5Screate(H5S_SCALAR);
    if (dataspace < 0) {
        printf("Error: Unable to create HDF5 dataspace for string attribute - %s.\n", name);
        return;
    }

    datatype = H5Tcopy(H5T_C_S1);
    if (datatype < 0) {
        printf("Error: Unable to create HDF5 datatype for string attribute - %s.\n", name);
        return;
    }
    H5Tset_size(datatype, strlen(data));
    H5Tset_strpad(datatype, H5T_STR_NULLTERM);
    H5Tset_cset(datatype, H5T_CSET_UTF8);

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0) {
        printf("Error: Unable to create HDF5 Attribute for string attribute - %s.\n", name);
        return;
    }

    if (H5Awrite(attribute, datatype, data) < 0) {
        printf("Warning: Unable to write HDF5 Attribute for string attribute - %s.\n", name);
    }
    // Add size to fileSize
    fileSize += STRBUFFSIZE;

    if (H5Sclose(dataspace) < 0) {
        printf("Warning: Unable to close HDF5 dataspace for string attribute - %s\n", name);
    }
    if (H5Tclose(datatype) < 0) {
        printf("Warning: Unable to close HDF5 datatype for string attribute - %s\n", name);
    }
    if (H5Aclose(attribute) < 0) {
        printf("Warning: Unable to close HDF5 attribute for string attribute - %s\n", name);
    }
}

/**
 * Create a singular numerical attribute attached to the given group
 */
void createNumAttribute(hid_t group, const char *name, hid_t dtype, unsigned long long data) {
    hid_t datatype, dataspace; /* handles */
    hid_t attribute;
    unsigned long long attr_data[1];
    attr_data[0] = data;

    dataspace = H5Screate(H5S_SCALAR);
    if (dataspace < 0) {
        printf("Error: Unable to create HDF5 dataspace for numerical attribute - %s.\n", name);
        return;
    }

    datatype = H5Tcopy(dtype);
    if (datatype < 0) {
        printf("Error: Unable to create HDF5 datatype for numerical attribute - %s.\n", name);
        return;
    }

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0) {
        printf("Error: Unable to create HDF5 Attribute for numerical attribute - %s.\n", name);
        return;
    }

    if (H5Awrite(attribute, dtype, attr_data) < 0) {
        printf("Warning: Unable to write HDF5 Attribute for numerical attribute - %s.\n", name);
    }
    fileSize += 16;

    if (H5Sclose(dataspace) < 0) {
        printf("Warning: Unable to close HDF5 dataspace for numerical attribute - %s\n", name);
    }
    if (H5Tclose(datatype) < 0) {
        printf("Warning: Unable to close HDF5 datatype for numerical attribute - %s\n", name);
    }
    if (H5Aclose(attribute) < 0) {
        printf("Warning: Unable to close HDF5 attribute for numerical attribute - %s\n", name);
    }
}

/**
 * Create a singular float attribute attached to the given group
 */
void createFloatAttribute(hid_t group, const char *name, float data) {
    hid_t datatype, dataspace; /* handles */
    hid_t attribute;
    float attr_data[1];
    attr_data[0] = data;

    dataspace = H5Screate(H5S_SCALAR);
    if (dataspace < 0) {
        printf("Error: Unable to create HDF5 dataspace for floating point attribute - %s.\n", name);
        return;
    }

    datatype = H5Tcopy(H5T_NATIVE_FLOAT);
    if (datatype < 0) {
        printf("Error: Unable to create HDF5 datatype for floating point attribute - %s.\n", name);
        return;
    }

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0) {
        printf("Error: Unable to create HDF5 Attribute for floating point attribute - %s.\n", name);
        return;
    }

    if (H5Awrite(attribute, H5T_NATIVE_FLOAT, attr_data) < 0) {
        printf("Warning: Unable to write HDF5 Attribute for floating point attribute - %s.\n", name);
    }
    fileSize += 32;

    if (H5Sclose(dataspace) < 0) {
        printf("Warning: Unable to close HDF5 dataspace for floating point attribute - %s.\n", name);
    }
    if (H5Tclose(datatype) < 0) {
        printf("Warning: Unable to close HDF5 datatype for floating point attribute - %s.\n", name);
    }
    if (H5Aclose(attribute) < 0) {
        printf("Warning: Unable to close HDF5 attribute for floating point attribute - %s.\n", name);
    }
}

/**
 * Create a singular double attribute attached to the given group.
 */
void createDoubleAttribute(hid_t group, const char *name, double data) {
    hid_t datatype, dataspace; /* handles */
    hid_t attribute;
    double attr_data[1];
    attr_data[0] = data;

    dataspace = H5Screate(H5S_SCALAR);
    if (dataspace < 0) {
        printf("Error: Unable to create HDF5 dataspace for double precision attribute - %s.\n", name);
        return;
    }

    datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
    if (datatype < 0) {
        printf("Error: Unable to create HDF5 datatype for double precision attribute - %s.\n", name);
        return;
    }

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0) {
        printf("Error: Unable to create HDF5 Attribute for double precision attribute - %s.\n", name);
        return;
    }

    if (H5Awrite(attribute, H5T_NATIVE_DOUBLE, attr_data) < 0) {
        printf("Warning: Unable to write HDF5 Attribute for double precision attribute - %s.\n", name);
    }
    fileSize += 64;

    if (H5Sclose(dataspace) < 0) {
        printf("Warning: Unable to close HDF5 dataspace for double precision attribute - %s.\n", name);
    }
    if (H5Tclose(datatype) < 0) {
        printf("Warning: Unable to close HDF5 datatype for double precision attribute - %s.", name);
    }
    if (H5Aclose(attribute) < 0) {
        printf("Warning: Unable to close HDF5 attribute for double precision attribute - %s.\n", name);
    }
}

/**
 * Send Redis Command
 * @return The redis reply from HIREDIS
 */
redisReply *sendHSETRedisCommand(redisContext *redisServer, const char *HSETName, const char *Value) {
    redisReply *reply = (redisReply *)redisCommand(redisServer, "HGET %s %s", HSETName, Value);
    if (reply->type != REDIS_REPLY_STRING) {
        printf("Warning: Redis was unable to get replay with the command - HGET %s %s\n", HSETName, Value);
        freeReplyObject(reply);
        return NULL;
    }
    return reply;
}

/**
 * Send Redis Command
 * @return The redis reply from HIREDIS
 */
redisReply *sendHSETRedisCommand(redisContext *redisServer, int HSETName, const char *Value) {
    redisReply *reply = (redisReply *)redisCommand(redisServer, "HGET %i %s", HSETName, Value);
    if (reply->type != REDIS_REPLY_STRING) {
        printf("Warning: Redis was unable to get replay with the command - HGET %i %s\n", HSETName, Value);
        freeReplyObject(reply);
        return NULL;
    }
    return reply;
}

/**
 * Initialize a new file given a name and time.
 */
fileIDs_t *createNewFile(char *fileName, char *currTime) {
    fileIDs_t *newfile = (fileIDs_t *)malloc(sizeof(struct fileIDs));
    if (newfile == NULL) {
        printf("Error: Unable to malloc space for new file.\n");
        exit(1);
    }

    newfile->file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    createStrAttribute(newfile->file, "dateCreated", currTime);

    FILE *fp;
    char ntpout[1035];
    fp = popen("ntpstat", "r");
    size_t size;
    if (fp != NULL){
        size = fread(ntpout, sizeof(char), 1035, fp);//fgets(out, 1035, fp);
        if (size){
            ntpout[size] = '\0';
            createStrAttribute(newfile->file, "ntpstat", ntpout);
        }
    }

    //;createStrAttribute(newfile->file, "ntpstat", system("ntpstat"));
    newfile->bit16IMGData = H5Gcreate(newfile->file, "/bit16IMGData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit8IMGData = H5Gcreate(newfile->file, "/bit8IMGData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->PHData = H5Gcreate(newfile->file, "/PHData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->ShortTransient = H5Gcreate(newfile->file, "/ShortTransient", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit16HCData = H5Gcreate(newfile->file, "/bit16HCData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit8HCData = H5Gcreate(newfile->file, "/bit8HCData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->DynamicMeta = H5Gcreate(newfile->file, "/DynamicMeta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->StaticMeta = H5Gcreate(newfile->file, "/StaticMeta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (newfile->file < 0 || newfile->bit16IMGData < 0 || newfile->bit8IMGData < 0 ||
        newfile->PHData < 0 || newfile->ShortTransient < 0 || newfile->bit16HCData < 0 ||
        newfile->bit8HCData < 0 || newfile->DynamicMeta < 0 || newfile->StaticMeta < 0) {
        printf("Error in creating HD5f file\n");
        exit(1);
    } else {
        printf("Created new file: %s\n", fileName);
    }

    return newfile;
}

/**
 * Create new quabo tables within the HDF5 file located at the group.
 */
void createQuaboTables(hid_t group, modulePairFile_t *module) {

    HKPackets_t HK_data;
    char tableName[50];
    char tableTitle[50];
    for (int i = 0; i < QUABOPERMODULE; i++) {
        sprintf(tableName, HK_TABLENAME_FORAMT, module->mod1Name, i);
        sprintf(tableTitle, HK_TABLETITLE_FORMAT, module->mod1Name, i);

        if (H5TBmake_table(tableTitle, group, tableName, HKFIELDS, 0,
                           HK_dst_size, HK_field_names, HK_dst_offset, HK_field_types,
                           100, NULL, 0, &HK_data) < 0) {
            printf("Error: Unable to create quabo tables in HDF5 file.\n");
            exit(1);
        }
    }

    for (int i = 0; i < QUABOPERMODULE; i++) {
        sprintf(tableName, HK_TABLENAME_FORAMT, module->mod2Name, i);
        sprintf(tableTitle, HK_TABLETITLE_FORMAT, module->mod2Name, i);

        if (H5TBmake_table(tableTitle, group, tableName, HKFIELDS, 0,
                           HK_dst_size, HK_field_names, HK_dst_offset, HK_field_types,
                           100, NULL, 0, &HK_data) < 0) {
            printf("Error: Unable to create quabo tables in HDF5 file.\n");
            exit(1);
        }
    }
}

/**
 * Create Module Pair Pointers from the config file
 */
void create_ModPair(fileIDs_t *currFile, modulePairFile_t **moduleFileInd, modulePairFile_t *moduleLinkEnd) {
    //Initializing the Module Pairing using the config file given
    FILE *modConfig_file = fopen(CONFIGFILE, "r");
    char fbuf[100];
    char cbuf;
    unsigned int mod1Name;
    unsigned int mod2Name;

    if (modConfig_file == NULL) {
        perror("Error Opening Config File\n");
        exit(1);
    }
    cbuf = getc(modConfig_file);
    char moduleName[50];

    while (cbuf != EOF) {
        ungetc(cbuf, modConfig_file);
        if (cbuf != '#') {
            if (fscanf(modConfig_file, "%u %u\n", &mod1Name, &mod2Name) == 2) {
                if (moduleFileInd[mod1Name] == NULL && moduleFileInd[mod2Name] == NULL) {

                    sprintf(moduleName, MODULEPAIR_FORMAT, mod1Name, mod2Name);

                    moduleFileInd[mod1Name] = moduleFileInd[mod2Name] = moduleLinkEnd->next_modulePairFile = modulePairFile_t_new(currFile, mod1Name, mod2Name);

                    moduleLinkEnd = moduleLinkEnd->next_modulePairFile;

                    createQuaboTables(currFile->DynamicMeta, moduleLinkEnd);

                    printf("Created Module Pair: %u.%u-%u and %u.%u-%u\n",
                           (unsigned int)(mod1Name << 2) / 0x100, (mod1Name << 2) % 0x100, ((mod1Name << 2) % 0x100) + 3,
                           (mod2Name << 2) / 0x100, (mod2Name << 2) % 0x100, ((mod2Name << 2) % 0x100) + 3);
                }
            }
        } else {
            if (fgets(fbuf, 100, modConfig_file) == NULL) {
                break;
            }
        }
        cbuf = getc(modConfig_file);
    }

    if (fclose(modConfig_file) == EOF) {
        printf("Warning: Unable to close module configuration file.\n");
    }
}

/**
 * 
 * REDIS METHODS
 * 
 */

/**
 * Get and store the GPS Supplimentary data in the HDF5 file.
 */
void get_storeGPSSupp(redisContext *redisServer, hid_t group) {
    redisReply *reply;

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "RECEIVERMODE");
    if (reply != NULL) {
        createStrAttribute(group, "RECEIVERMODE", reply->str);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DISCIPLININGMODE");
    if (reply != NULL) {
        createStrAttribute(group, "DISCIPLININGMODE", reply->str);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "SELFSURVEYPROGRESS");
    if (reply != NULL) {
        createNumAttribute(group, "SELFSURVEYPROGRESS", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "HOLDOVERDURATION");
    if (reply != NULL) {
        createNumAttribute(group, "HOLDOVERDURATION", H5T_STD_U32LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DACatRail");
    if (reply != NULL) {
        createNumAttribute(group, "DACatRail", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DACnearRail");
    if (reply != NULL) {
        createNumAttribute(group, "DACnearRail", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "AntennaOpen");
    if (reply != NULL) {
        createNumAttribute(group, "AntennaOpen", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "AntennaShorted");
    if (reply != NULL) {
        createNumAttribute(group, "AntennaShorted", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "NotTrackingSatellites");
    if (reply != NULL) {
        createNumAttribute(group, "NotTrackingSatellites", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "NotDiscipliningOscillator");
    if (reply != NULL) {
        createNumAttribute(group, "NotDiscipliningOscillator", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "SurveyInProgress");
    if (reply != NULL) {
        createNumAttribute(group, "SurveyInProgress", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "NoStoredPosition");
    if (reply != NULL) {
        createNumAttribute(group, "NoStoredPosition", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "LeapSecondPending");
    if (reply != NULL) {
        createNumAttribute(group, "LeapSecondPending", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "InTestMode");
    if (reply != NULL) {
        createNumAttribute(group, "InTestMode", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "PositionIsQuestionable");
    if (reply != NULL) {
        createNumAttribute(group, "PositionIsQuestionable", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "EEPROMCorrupt");
    if (reply != NULL) {
        createNumAttribute(group, "EEPROMCorrupt", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "AlmanacNotComplete");
    if (reply != NULL) {
        createNumAttribute(group, "AlmanacNotComplete", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "PPSNotGenerated");
    if (reply != NULL) {
        createNumAttribute(group, "PPSNotGenerated", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "GPSDECODINGSTATUS");
    if (reply != NULL) {
        createNumAttribute(group, "GPSDECODINGSTATUS", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DISCIPLININGACTIVITY");
    if (reply != NULL) {
        createNumAttribute(group, "DISCIPLININGACTIVITY", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "PPSOFFSET");
    if (reply != NULL) {
        createFloatAttribute(group, "PPSOFFSET", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "CLOCKOFFSET");
    if (reply != NULL) {
        createFloatAttribute(group, "CLOCKOFFSET", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DACVALUE");
    if (reply != NULL) {
        createNumAttribute(group, "DACVALUE", H5T_STD_U32LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DACVOLTAGE");
    if (reply != NULL) {
        createFloatAttribute(group, "DACVOLTAGE", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "TEMPERATURE");
    if (reply != NULL) {
        createFloatAttribute(group, "TEMPERATURE", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "LATITUDE");
    if (reply != NULL) {
        createDoubleAttribute(group, "LATITUDE", strtod(reply->str, NULL));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "LONGITUDE");
    if (reply != NULL) {
        createDoubleAttribute(group, "LONGITUDE", strtod(reply->str, NULL));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "ALTITUDE");
    if (reply != NULL) {
        createDoubleAttribute(group, "ALTITUDE", strtod(reply->str, NULL));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "PPSQUANTIZATIONERROR");
    if (reply != NULL) {
        createFloatAttribute(group, "PPSQUANTIZATIONERROR", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }
}

/**
 * Get and store the White Rabbit Switch data into HDF5 file.
 */
void get_storeWR(redisContext *redisServer, hid_t group) {
    redisReply *reply = (redisReply *)redisCommand(redisServer, "HGETALL %s", WRSWITCHNAME);
    if (reply->type != REDIS_REPLY_ARRAY) {
        printf("Warning: Unable to get WR Swtich Values from Redis. Skipping WR Data.\n");
        return;
    }

    for (int i = 0; i < reply->elements; i = i + 2) {
        createNumAttribute(group, reply->element[i]->str, H5T_STD_U8LE, strtoll(reply->element[i + 1]->str, NULL, 10));
    }
    freeReplyObject(reply);
}

/**
 * Get the GPS data from the Redis Server.
 */
void fetchGPSdata(GPSPackets_t *GPS, redisContext *redisServer) {
    redisReply *reply;

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "GPSTIME");
    if (reply != NULL) {
        strcpy(GPS->GPSTIME, reply->str);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TOW");
    if (reply != NULL) {
        GPS->TOW = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "WEEKNUMBER");
    if (reply != NULL) {
        GPS->WEEKNUMBER = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "UTCOFFSET");
    if (reply != NULL) {
        GPS->UTCOFFSET = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TIMEFLAG");
    if (reply != NULL) {
        strcpy(GPS->TIMEFLAG, reply->str);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "PPSFLAG");
    if (reply != NULL) {
        strcpy(GPS->PPSFLAG, reply->str);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TIMESET");
    if (reply != NULL) {
        GPS->TIMESET = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "UTCINFO");
    if (reply != NULL) {
        GPS->UTCINFO = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TIMEFROMGPS");
    if (reply != NULL) {
        GPS->TIMEFROMGPS = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TV_UTC");
    if (reply != NULL) {
        strcpy(GPS->TV_UTC, reply->str);
        freeReplyObject(reply);
    }
}

/**
 * Fetch the Housekeeping data from the Redis database for the given boardloc or quabo id.
 */
void fetchHKdata(HKPackets_t *HK, uint16_t BOARDLOC, redisContext *redisServer) {
    redisReply *reply;

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "SYSTIME");
    if (reply != NULL) {
        strcpy(HK->SYSTIME, reply->str);
        freeReplyObject(reply);
    }
    else {
        strcpy(HK->SYSTIME, "");
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "BOARDLOC");
    if (reply != NULL) {
        HK->BOARDLOC = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    else {
        HK->BOARDLOC = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVMON0");
    if (reply != NULL) {
        HK->HVMON0 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->HVMON0 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVMON1");
    if (reply != NULL) {
        HK->HVMON1 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->HVMON1 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVMON2");
    if (reply != NULL) {
        HK->HVMON2 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->HVMON2 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVMON3");
    if (reply != NULL) {
        HK->HVMON3 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->HVMON3 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVIMON0");
    if (reply != NULL) {
        HK->HVIMON0 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->HVIMON0 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVIMON1");
    if (reply != NULL) {
        HK->HVIMON1 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->HVIMON1 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVIMON2");
    if (reply != NULL) {
        HK->HVIMON2 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->HVIMON2 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVIMON3");
    if (reply != NULL) {
        HK->HVIMON3 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->HVIMON3 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "RAWHVMON");
    if (reply != NULL) {
        HK->RAWHVMON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->RAWHVMON = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "V12MON");
    if (reply != NULL) {
        HK->V12MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->V12MON = 0;
    }
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "V18MON");
    if (reply != NULL) {
        HK->V18MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->V18MON = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "V33MON");
    if (reply != NULL) {
        HK->V33MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->V33MON = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "V37MON");
    if (reply != NULL) {
        HK->V37MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->V37MON = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "I10MON");
    if (reply != NULL) {
        HK->I10MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->I10MON = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "I18MON");
    if (reply != NULL) {
        HK->I18MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->I18MON = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "I33MON");
    if (reply != NULL) {
        HK->I33MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->I33MON = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "TEMP1");
    if (reply != NULL) {
        HK->TEMP1 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->TEMP1 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "TEMP2");
    if (reply != NULL) {
        HK->TEMP2 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->TEMP2 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "VCCINT");
    if (reply != NULL) {
        HK->VCCINT = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->VCCINT = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "VCCAUX");
    if (reply != NULL) {
        HK->VCCAUX = strtof(reply->str, NULL);
        freeReplyObject(reply);
    }
    else {
        HK->VCCAUX = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "UID");
    if (reply != NULL) {
        HK->UID = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    else {
        HK->UID = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "SHUTTER_STATUS");
    if (reply != NULL) {
        HK->SHUTTER_STATUS = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    else {
        HK->SHUTTER_STATUS = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "LIGHT_SENSOR_STATUS");
    if (reply != NULL) {
        HK->LIGHT_STATUS = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    else {
        HK->LIGHT_STATUS = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "FWID0");
    if (reply != NULL) {
        HK->FWID0 = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    else {
        HK->FWID0 = 0;
    }

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "FWID1");
    if (reply != NULL) {
        HK->FWID1 = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    else {
        HK->FWID1 = 0;
    }
}

/**
 * Check and store Static data to the HDF5 file.
 */
void getStaticRedisData(redisContext *redisServer, hid_t staticMeta) {
    hid_t GPSgroup, WRgroup;
    GPSgroup = H5Gcreate(staticMeta, GPSSUPPNAME, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (GPSgroup < 0) {
        printf("Error: Unable to create GPS group in HDF5 file.\n");
        exit(1);
    }
    WRgroup = H5Gcreate(staticMeta, WRSWITCHNAME, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (WRgroup < 0) {
        printf("Error: Unable to create White Rabbit group in HDF5 file.\n");
        exit(1);
    }

    get_storeGPSSupp(redisServer, GPSgroup);
    get_storeWR(redisServer, WRgroup);

    if (H5Gclose(GPSgroup) < 0) {
        printf("Warning: Unable to close GPS HDF5 Group\n");
    }
    if (H5Gclose(WRgroup) < 0) {
        printf("Warning: Unable to close WR HDF5 Group\n");
    }
}

/**
 * Check if housekeeping data for the module pair has been updated and if so get and store it in the HDF5 file.
 */
void check_storeHK(redisContext *redisServer, modulePairFile_t *modFileHead, hid_t dynamicMeta) {
    HKPackets_t *HKdata = (HKPackets_t *)malloc(sizeof(HKPackets));
    if (HKdata == NULL) {
        printf("Error: Unable to malloc space for House Keeping Object.\n");
        exit(1);
    }
    modulePairFile_t *currentModFile;
    redisReply *reply;
    uint16_t BOARDLOC;
    char tableName[50];

    currentModFile = modFileHead;

    while (currentModFile != NULL) {

        //Updating all the Quabos from Module 1
        BOARDLOC = (currentModFile->mod1Name << 2) & 0xfffc;

        for (int i = 0; i < 4; i++) {
            reply = (redisReply *)redisCommand(redisServer, "HGET UPDATED %u", BOARDLOC);

            if (strtol(reply->str, NULL, 10)) {
                freeReplyObject(reply);

                fetchHKdata(HKdata, BOARDLOC, redisServer);
                sprintf(tableName, HK_TABLENAME_FORAMT, currentModFile->mod1Name, i);
                if (H5TBappend_records(dynamicMeta, tableName, 1, HK_dst_size, HK_dst_offset, HK_dst_sizes, HKdata) < 0) {
                    printf("Warning: Unable to write HK Data for module %i-%i\n", currentModFile->mod1Name, i);
                }
                fileSize += HKDATASIZE;

                reply = (redisReply *)redisCommand(redisServer, "HSET UPDATED %u 0", BOARDLOC);
            }

            freeReplyObject(reply);
            BOARDLOC++;
        }

        if (currentModFile->mod2Name != -1) {
            //Updating all the Quabos from Module 2
            BOARDLOC = (currentModFile->mod2Name << 2) & 0xfffc;

            for (int i = 0; i < 4; i++) {
                reply = (redisReply *)redisCommand(redisServer, "HGET UPDATED %u", BOARDLOC);

                if (strtol(reply->str, NULL, 10)) {
                    freeReplyObject(reply);

                    fetchHKdata(HKdata, BOARDLOC, redisServer);
                    sprintf(tableName, HK_TABLENAME_FORAMT, currentModFile->mod2Name, i);
                    if (H5TBappend_records(dynamicMeta, tableName, 1, HK_dst_size, HK_dst_offset, HK_dst_sizes, HKdata) < 0) {
                        printf("Warning: Unable to write HK Data for module %i-%i\n", currentModFile->mod1Name, i);
                    }

                    fileSize += HKDATASIZE;

                    reply = (redisReply *)redisCommand(redisServer, "HSET UPDATED %u 0", BOARDLOC);
                }

                freeReplyObject(reply);
                BOARDLOC++;
            }
        }

        //Update to Next Module
        currentModFile = currentModFile->next_modulePairFile;
    }

    free(HKdata);
}

/**
 * Check if the GPS Primary data have been updated and if so store the GPS Primary data in the HDF5 file.
 */
void check_storeGPS(redisContext *redisServer, hid_t group) {
    GPSPackets_t *GPSdata = (GPSPackets_t *)malloc(sizeof(GPSPackets));
    if (GPSdata == NULL) {
        printf("Warning: Unable to malloc space for GPS Object. Skipping GPS Data storage\n");
        return;
    }
    redisReply *reply = (redisReply *)redisCommand(redisServer, "HGET UPDATED %s", GPSPRIMNAME);
    if (reply->type != REDIS_REPLY_STRING) {
        printf("Warning: Unable to get GPS's UPDATED Flag from Redis. Skipping GPS Data.\n");
        return;
    }

    if (strtol(reply->str, NULL, 10)) {
        freeReplyObject(reply);

        fetchGPSdata(GPSdata, redisServer);
        if (H5TBappend_records(group, GPSPRIMNAME, 1, GPS_dst_size, GPS_dst_offset, GPS_dst_sizes, GPSdata) < 0) {
            printf("Warning: Unable to append GPS data to table in HDF5 file.\n");
        }

        reply = (redisReply *)redisCommand(redisServer, "HSET UPDATED %s 0", GPSPRIMNAME);

        if (reply->type != REDIS_REPLY_INTEGER) {
            printf("Warning: Unable to set GPS's UPDATED Flag from Redis.\n");
        }
    }

    freeReplyObject(reply);
    free(GPSdata);
}

/**
 * Check and store Dynamic data to the HDF5 file.
 */
void getDynamicRedisData(redisContext *redisServer, modulePairFile_t *modFileHead, hid_t dynamicMeta) {
    check_storeHK(redisServer, modFileHead, dynamicMeta);
    check_storeGPS(redisServer, dynamicMeta);
}

/**
 * Create new GPS tables within the HDF5 files located at the group
 */
void createGPSTable(hid_t group) {
    GPSPackets_t GPS_data;

    if (H5TBmake_table(GPSPRIMNAME, group, GPSPRIMNAME, GPSFIELDS, 0,
                       GPS_dst_size, GPS_field_names, GPS_dst_offset, GPS_field_types,
                       100, NULL, 0, &GPS_data) < 0) {
        printf("Unable to create GPS Table for HDF5 file\n");
        exit(1);
    }
}

/**
 * Create new White Rabbit Switch tables within the HDF5 files located at the group.
 */
void createWRTable() {
}

/**
 * Inialize the metadata resources such as GPS and WR tables.
 */
void createDMetaResources(hid_t group) {
    createGPSTable(group);
    createWRTable();
}

/**
 * Initilzing the HDF5 file based on the current file_naming format.
 */
fileIDs_t *HDF5file_init() {
    time_t t = time(NULL);
    struct tm tm = *gmtime(&t);
    char currTime[STRBUFFSIZE + 20];
    char fileName[STRBUFFSIZE + 20];

    //Making the directory for where the data files are stored
    sprintf(fileName, "%s%04i/", saveLocation, (tm.tm_year + 1900));
    mkdir(fileName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    sprintf(fileName, "%s%04i/%04i%02i%02i/", saveLocation, (tm.tm_year + 1900), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
    mkdir(fileName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    sprintf(currTime, TIME_FORMAT, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    sprintf(fileName + strlen(fileName), H5FILE_NAME_FORMAT, OBSERVATORY, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

    if (access(fileName, F_OK) != -1) {
        printf("Error: Unable to access file location - %s", fileName);
        exit(0);
    }

    fileIDs_t* new_file = createNewFile(fileName, currTime);

    createDMetaResources(new_file->DynamicMeta);

    return new_file;
}

fileIDs_t* reInitHDF5File(fileIDs_t* oldFile, modulePairFile_t* moduleFileListBegin, modulePairFile_t* moduleFileListEnd, modulePairFile_t** moduleFileIndex){
    fileIDs_t* new_file = HDF5file_init();
    modulePairFile_t* modFileEndptr = moduleFileListBegin;
    modulePairFile_t* modFileoldHeadptr = moduleFileListBegin->next_modulePairFile;
    modulePairFile_t* modFileToFree;
    while (modFileoldHeadptr){
        //Close old ModulePair
        if (modFileoldHeadptr->bit16Dataset >= 0){
            H5Dclose(modFileoldHeadptr->bit16Dataset);
            H5Dclose(modFileoldHeadptr->bit16pktNum);
            H5Dclose(modFileoldHeadptr->bit16pktNSEC);
            H5Dclose(modFileoldHeadptr->bit16tv_sec);
            H5Dclose(modFileoldHeadptr->bit16tv_usec);
            H5Dclose(modFileoldHeadptr->bit16status);
        }
        H5Gclose(modFileoldHeadptr->bit16IMGGroup);
        if (modFileoldHeadptr->bit8Dataset >= 0){
            H5Dclose(modFileoldHeadptr->bit8Dataset);
            H5Dclose(modFileoldHeadptr->bit8pktNum);
            H5Dclose(modFileoldHeadptr->bit8pktNSEC);
            H5Dclose(modFileoldHeadptr->bit8tv_sec);
            H5Dclose(modFileoldHeadptr->bit8tv_usec);
            H5Dclose(modFileoldHeadptr->bit8status);
        }
        H5Gclose(modFileoldHeadptr->bit8IMGGroup);
        if (modFileoldHeadptr->PHDataset >= 0){
            H5Dclose(modFileoldHeadptr->PHDataset);
            H5Dclose(modFileoldHeadptr->PHpktNum);
            H5Dclose(modFileoldHeadptr->PHpktNSEC);
            H5Dclose(modFileoldHeadptr->PHtv_sec);
            H5Dclose(modFileoldHeadptr->PHtv_usec);
            H5Dclose(modFileoldHeadptr->PHmodNum);
            H5Dclose(modFileoldHeadptr->PHquaNum);
            H5Dclose(modFileoldHeadptr->PHpktUTC);
        }
        H5Gclose(modFileoldHeadptr->PHGroup);

        //Reinitate new ModFile Pairs
        moduleFileIndex[modFileoldHeadptr->mod1Name] = moduleFileIndex[modFileoldHeadptr->mod2Name] 
        = modFileEndptr->next_modulePairFile 
        = modulePairFile_t_new(new_file, modFileoldHeadptr->mod1Name, modFileoldHeadptr->mod2Name);
        createQuaboTables(new_file->DynamicMeta, modFileEndptr->next_modulePairFile);

        //Increment ptrs and free old object
        modFileEndptr = modFileEndptr->next_modulePairFile;
        modFileToFree = modFileoldHeadptr;
        modFileoldHeadptr = modFileoldHeadptr->next_modulePairFile;
        free(modFileToFree);
    }
    moduleFileListEnd = modFileEndptr;

    //Closing File Resources
    H5Gclose(oldFile->StaticMeta);
    H5Gclose(oldFile->DynamicMeta);
    H5Gclose(oldFile->bit16IMGData);
    H5Gclose(oldFile->bit8IMGData);
    H5Gclose(oldFile->PHData);
    H5Gclose(oldFile->ShortTransient);
    H5Gclose(oldFile->bit16HCData);
    H5Gclose(oldFile->bit8HCData);
    H5Fclose(oldFile->file);
    free(oldFile);

    return new_file;
}

static fileIDs_t *file;

static modulePairFile_t *moduleFileListBegin;
static modulePairFile_t *moduleFileListEnd;
static modulePairFile_t *moduleFileIndex[MODULEINDEXSIZE] = {NULL};

static redisContext *redisServer;

//Signal handeler to allow for hashpipe to exit gracfully and also to allow for creating of new files by command.
static int QUITSIG;

void QUIThandler(int signum) {
    QUITSIG = 1;
}

static int init(hashpipe_thread_args_t *args)
{
    H5Pset_chunk(creation_property, RANK, chunkDim);
    // Get info from status buffer if present
    hashpipe_status_t st = args->st;
    printf("\n\n-----------Start Setup of Output Thread--------------\n");
    sprintf(saveLocation, "./");
    hgets(st.buf, "SAVELOC", STRBUFFSIZE, saveLocation);
    if (saveLocation[strlen(saveLocation) - 1] != '/') {
        char endingSlash = '/';
        strncat(saveLocation, &endingSlash, 1);
        //saveLocation[strlen(saveLocation)] = '/';
    }
    printf("Save Location: %s\n", saveLocation);

    int maxSizeInput = 0;

    hgeti4(st.buf, "MAXFILESIZE", &maxSizeInput);
    maxFileSize = maxSizeInput * 2E6;

    /*Initialization of Redis Server Values*/
    printf("------------------SETTING UP REDIS ------------------\n");
    redisServer = redisConnect("127.0.0.1", 6379);
    int attempts = 0;
    while (redisServer != NULL && redisServer->err) {
        printf("Error: %s\n", redisServer->errstr);
        attempts++;
        if (attempts >= 12) {
            printf("Unable to connect to Redis.\n");
            exit(0);
        }
        printf("Attempting to reconnect in 5 seconds.\n");
        sleep(5);
        redisServer = redisConnect("127.0.0.1", 6379);
    }

    printf("Connected to Redis\n");
    redisReply *keysReply;
    redisReply *reply;
    // Uncomment following lines for redis servers with password
    // reply = redisCommand(redisServer, "AUTH password");
    // freeReplyObject(reply);

    printf("-----------Finished Setup of Output Thread-----------\n\n");    

    return 0;
}

static void *run(hashpipe_thread_args_t *args) {

    signal(SIGQUIT, QUIThandler);

    QUITSIG = 0;
    /* Initialization of HDF5 Values*/
    printf("\n-------------------SETTING UP HDF5 ------------------\n");

    file = HDF5file_init();
    moduleFileListBegin = modulePairFile_t_new(file, -1, -1, 0);
    moduleFileListEnd = moduleFileListBegin;
    create_ModPair(file, moduleFileIndex, moduleFileListEnd);

    getStaticRedisData(redisServer, file->StaticMeta);

    getDynamicRedisData(redisServer, moduleFileListBegin->next_modulePairFile, file->DynamicMeta);

    
    printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n");

    printf("---------------Running Output Thread-----------------\n\n");

    /*Initialization of HASHPIPE Values*/
    // Local aliases to shorten access to args fields
    // Our input buffer happens to be a demo1_ouput_databuf
    HSD_output_databuf_t *db = (HSD_output_databuf_t *)args->ibuf;
    hashpipe_status_t st = args->st;
    const char *status_key = args->thread_desc->skey;

    int rv;
    int block_idx = 0;
    uint64_t mcnt = 0;
    modulePairFile_t *currModPairFile;

    /* Main loop */
    while (run_threads()) {

        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "OUTBLKIN", block_idx);
        hputi8(st.buf, "OUTMCNT", mcnt);
        hputs(st.buf, status_key, "waiting");
        hashpipe_status_unlock_safe(&st);

        //Wait for the output buffer to be free
        while ((rv = HSD_output_databuf_wait_filled(db, block_idx)) != HASHPIPE_OK)
        {
            if (rv == HASHPIPE_TIMEOUT)
            {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked");
                hashpipe_status_unlock_safe(&st);
                continue;
            }
            else
            {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
                pthread_exit(NULL);
                break;
            }
        }

        // Mark the buffer as processing
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "processing");
        hashpipe_status_unlock_safe(&st);

        getDynamicRedisData(redisServer, moduleFileListBegin->next_modulePairFile, file->DynamicMeta);
        for (int i = 0; i < db->block[block_idx].header.stream_block_size; i++) {
            if (moduleFileIndex[db->block[block_idx].header.modNum[i * 2]]) {
                currModPairFile = moduleFileIndex[db->block[block_idx].header.modNum[i * 2]];
                //printf("Mod1 Number is :%i Acqmode: %i\n", db->block[block_idx].header.modNum[i*2],
                //                                            db->block[block_idx].header.acqmode[i]);
            } else if (moduleFileIndex[db->block[block_idx].header.modNum[(i * 2) + 1]]) {
                currModPairFile = moduleFileIndex[db->block[block_idx].header.modNum[(i * 2) + 1]];
                //printf("Mod2 Number is :%i Acqmode: %i\n", db->block[block_idx].header.modNum[(i*2)+1],
                //                                            db->block[block_idx].header.acqmode[i]);
            }

            if (db->block[block_idx].header.acqmode[i] == 16) {
                #ifdef TEST_MODE
                    printf("Dataset Key: %i, ModulePair Index: %u\n", currModPairFile->bit16Dataset, currModPairFile->bit16ModPairIndex);
                #endif
                if (currModPairFile->bit16ModPairIndex >= (int)PKTPERDATASET) {
                    #ifdef TEST_MODE
                        printf("CreatingDataset\n");
                    #endif
                    create_ModPair_Dataset(currModPairFile, db->block[block_idx].header.acqmode[i]);
                }
                write_Dataset(currModPairFile, &(db->block[block_idx]), i);
                currModPairFile->bit16ModPairIndex += 1;
                fileSize += MODPAIRDATASIZE;

            } else if (db->block[block_idx].header.acqmode[i] == 8) {
                #ifdef TEST_MODE
                    printf("Dataset Key: %i, ModulePair Index: %u\n", currModPairFile->bit8Dataset, currModPairFile->bit8ModPairIndex);
                #endif
                if (currModPairFile->bit8ModPairIndex >= (int)PKTPERDATASET) {
                    create_ModPair_Dataset(currModPairFile, db->block[block_idx].header.acqmode[i]);
                }
                write_Dataset(currModPairFile, &(db->block[block_idx]), i);
                currModPairFile->bit8ModPairIndex += 1;
                fileSize += MODPAIRDATASIZE;
            }
        }


        for (int i = 0; i < db->block[block_idx].header.coinc_block_size; i++) {
            currModPairFile = moduleFileIndex[db->block[block_idx].header.coin_modNum[i]];

            if (currModPairFile->PHModPairIndex >= (int)PKTPERDATASET){
                create_ModPair_Dataset(currModPairFile, 0);
            }
            write_PHDataset(currModPairFile, &(db->block[block_idx]), i);
            currModPairFile->PHModPairIndex += 1;
        }

        if (QUITSIG || fileSize > maxFileSize) {
            printf("-----Start Reinitializing all File Resources----\n");
            file = reInitHDF5File(file, moduleFileListBegin, moduleFileListEnd, moduleFileIndex);
            getStaticRedisData(redisServer, file->StaticMeta);
            printf("-----Reinitializing File Resources Complete----\n");
            printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n\n");
            fileSize = 0;
            QUITSIG = 0;
        }

        //TODO check mcnt
        if (db->block[block_idx].header.INTSIG) {
            //closeAllResources();
            printf("OUTPUT_THREAD Ended\n");
            break;
        }

        HSD_output_databuf_set_free(db, block_idx);
        block_idx = (block_idx + 1) % db->header.n_block;
        mcnt++;

        /* Term conditions */

        //Will exit if thread has been cancelled
        pthread_testcancel();
    }

    printf("Returned Output_thread\n");
    return THREAD_OK;
}

/**
 * Sets the functions and buffers for this thread
 */
static hashpipe_thread_desc_t HSD_output_thread = {
    name : "HSD_output_thread",
    skey : "OUTSTAT",
    init : init,
    run : run,
    ibuf_desc : {HSD_output_databuf_create},
    obuf_desc : {NULL}
};

static __attribute__((constructor)) void ctor()
{
    register_hashpipe_thread(&HSD_output_thread);
}
