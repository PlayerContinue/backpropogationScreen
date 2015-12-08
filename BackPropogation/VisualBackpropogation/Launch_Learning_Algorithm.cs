using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

using VisualBackPropogation.Helper;
namespace VisualBackPropogation
{
    class Launch_Learning_Algorithm
    {
        private Process Learning_Algorithm;
        private ServerPipe Communication_Pipe;
        private Thread ServerThread;
        private string Application_Location;
        public Launch_Learning_Algorithm()
        {
       
        }

        public void Launch(string pipe_name,string settings_loc, string application_location){

            this.Learning_Algorithm = startProcessWithOutput("\"" + application_location + "\"", "\"" + settings_loc + "\" 1 1");
            Communication_Pipe = new ServerPipe();
            ServerThread = new Thread(() => Communication_Pipe.ThreadStartServer(pipe_name +"_OUT"));

            ServerThread.Start();
        }

        private Process startProcessWithOutput(string command, string args="", bool showWindow=true)
        {
            Process p = new Process();
            p.StartInfo = new ProcessStartInfo(command, args);
            p.StartInfo.RedirectStandardOutput = false;
            p.StartInfo.RedirectStandardError = false;
            p.StartInfo.UseShellExecute = true;
            p.StartInfo.CreateNoWindow = !showWindow;
            //p.ErrorDataReceived += (s, a) => addLogLine(a.Data);
            p.Start();
            //p.BeginErrorReadLine();

            return p;
        }

    }
}
