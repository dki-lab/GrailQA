import logging
import os
import sys
from allennlp.commands import main  # pylint: disable=wrong-import-position
# sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(pathname)s - %(lineno)s- %(message)s',
                    level=logging.DEBUG)

if __name__ == "__main__":
    main()
