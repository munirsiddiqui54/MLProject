import sys
from src.logger import logging


log=logging.getLogger(__name__)

def get_error_message(error,error_details_obj:sys):
    _,_,exc_tb=error_details_obj.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg="Error Occured. FILENAME: [{0}], LINE:{1}, ERROR: {2}".format(
        file_name,
        exc_tb.tb_lineno,
        error
    )
    return error_msg

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=get_error_message(error=error_message,error_details_obj=error_detail)
        log.error(self.error_message)
    
    def __str__(self):
        return self.error_message


    

