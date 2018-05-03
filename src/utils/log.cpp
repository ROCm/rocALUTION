#include "def.hpp"
#include "log.hpp"
#include "time_functions.hpp"
#include "../base/backend_manager.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

namespace rocalution {

void _rocalution_open_log_file(void) {

#ifdef LOG_FILE

  if (_get_backend_descriptor()->log_file != NULL) {
    _rocalution_close_log_file();
  }

#endif

  assert(_get_backend_descriptor()->log_file == NULL);

#ifdef LOG_FILE
  
  _get_backend_descriptor()->log_file = new std::ofstream;

  std::ostringstream str_double;
  str_double.precision(20);
  str_double << rocalution_time();
  std::string mid_name= str_double.str();
 
  std::ostringstream rank;
  rank << _get_backend_descriptor()->rank;
  std::string rank_name= rank.str();

  std::string str_name;

  str_name = "rocalution-rank-"
    + rank_name
    + "-"
    + mid_name 
    + ".log";
  
  _get_backend_descriptor()->log_file->open(str_name.c_str(), std::ios::out | std::ios::trunc);
  
#endif

}


void _rocalution_close_log_file(void) {

#ifdef LOG_FILE

  _get_backend_descriptor()->log_file->close();
  delete _get_backend_descriptor()->log_file;
  _get_backend_descriptor()->log_file = NULL;

#endif

}

}
