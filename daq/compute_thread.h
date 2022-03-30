#ifndef _COMPUTE_THREAD_H_
#define _COMPUTE_THREAD_H_

#include <string.h>

#include "databuf.h"

/**
 * Structure for monitoring module data held by the compute thread.
 * Software mode such as image integration and coincidence analysis
 * information should be stored in this structure.
 */
typedef struct module_data {
    uint32_t max_nanosec;
    uint32_t min_nanosec;
    uint8_t status;                 //Status is a freely used variable for either integration number or error code.
    module_header_t mod_head;
    uint8_t data[BYTES_PER_MODULE_FRAME];
    module_data(){
        this->max_nanosec = 0;
        this->min_nanosec = 0;
        this->status = 0;
    };
    int copy_to(module_data *mod_data) {
        mod_data->max_nanosec = this->max_nanosec;
        mod_data->min_nanosec = this->min_nanosec;
        mod_data->status = this->status;
        this->mod_head.copy_to(&(mod_data->mod_head));
        memcpy(mod_data->data, this->data, sizeof(uint8_t)*BYTES_PER_MODULE_FRAME);
    };
    int clear(){
        this->max_nanosec = 0;
        this->min_nanosec = 0;
        this->status = 0;
        this->mod_head.clear();
        memset(this->data, 0, sizeof(uint8_t)*BYTES_PER_MODULE_FRAME);
    };
    std::string toString(){
        return "status = " + std::to_string(this->status) +
                " max_nanosec = " + std::to_string(this->max_nanosec) +
                " min_nanosec = " + std::to_string(this->min_nanosec) +
                "\n" + mod_head.toString();
    };
    int equal_to(module_data *mod_data){
        if (this->max_nanosec != mod_data->max_nanosec
            || this->min_nanosec != mod_data->min_nanosec
            || this->status != mod_data->status){
            return 0;
        }
        if (!this->mod_head.equal_to(&(mod_data->mod_head))){
            return 0;
        }
        if (memcmp(this->data, mod_data->data, sizeof(uint8_t)*BYTES_PER_MODULE_FRAME) == 0){
            return 0;
        }
        return 1;
    }
} module_data_t;

#endif
