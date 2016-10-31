#include "config.h"

using namespace std;
param::param(const char *filename)
{
    using namespace std;
     // ifstream is used for reading files
    // We'll read from a file called Sample.dat
    ifstream inf("conf1.txt");
 
    // If we couldn't open the input file stream for reading
    if (!inf)
    {
        // Print an error and exit
        cerr << "Uh oh, Sample.dat could not be opened for reading!" << endl;
        exit(1);
    }
 
    // While there's still stuff left to read
    while (inf)
    {
        std::string strInput;
        inf >> strInput;
        if(strInput == "CREATE_VIDEO"){
            int i;
            inf >> i;
            if(i==1){
                CREATE_VIDEO = true;
            }
        }
        if(strInput == "IMAGE_FILE"){
            string filenamecpp;
            inf >> filenamecpp;
            IMAGE_FILE = new char [filenamecpp.length()+1];
            std::strcpy (IMAGE_FILE, filenamecpp.c_str());
        }
        if(strInput == "NUM_GENERATIONS"){
            inf >> NUM_GENERATIONS;
        }
        if(strInput == "SLOPE_THRESHOLD"){
            inf >> SLOPE_THRESHOLD;
            
        }
        if(strInput == "FORCED_DELAY"){
            inf >> FORCED_DELAY;
        }
        if(strInput == "RESOLUTION"){
            inf >> RESOLUTION;
        }
        if(strInput == "RUNAVGWID"){
            inf >> RUNAVGWID;
        }

        if(strInput == "NUM_NN"){
            inf >> NUM_NN;
        }
        if(strInput == "MAX_NODES"){
            inf >> MAX_NODES;
        }
        if(strInput == "NUM_DATA"){
            inf >> NUM_DATA;
        }
        if(strInput == "MAX_DATA_RESAMPLES"){
            inf >> MAX_DATA_RESAMPLES;
        }
        if(strInput == "START_NODES"){
            inf >> START_NODES;
        }
        if(strInput == "MAX_THREADS"){
            inf >> MAX_THREADS;
        }
        if(strInput == "SCALED_TUBE_THREADSHOLD"){
            inf >> SCALED_TUBE_THREADSHOLD;
        }
    }

}