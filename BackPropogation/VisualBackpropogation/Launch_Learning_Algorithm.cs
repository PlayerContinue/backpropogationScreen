using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VisualBackPropogation
{
    class Launch_Learning_Algorithm
    {
        private Process Learning_Algorithm;
        public Launch_Learning_Algorithm()
        {

        }

        public void Launch(){

            this.Learning_Algorithm = startProcessWithOutput("../../../DebugWithDebug/BackPropogation.exe","\"C:\\Users\\dgree\\Documents\\GitHub\\backpropogationScreen\\BackPropogation\\BackPropogation\\settings.txt\" 1 1");
        
        }

        private Process startProcessWithOutput(string command, string args="", bool showWindow=true)
        {
            Process p = new Process();
            p.StartInfo = new ProcessStartInfo(command, args);
            p.StartInfo.RedirectStandardOutput = false;
            p.StartInfo.RedirectStandardError = true;
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.CreateNoWindow = !showWindow;
            //p.ErrorDataReceived += (s, a) => addLogLine(a.Data);
            p.Start();
            p.BeginErrorReadLine();

            return p;
        }

    }
}
