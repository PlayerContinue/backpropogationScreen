using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.IO.Pipes;
using System.Threading;

namespace VisualBackPropogation.Helper
{
    class ServerPipe
    {

        public void ThreadStartServer(string pipe_name)
        {
            ConsoleManager.Show();

            // Create a name pipe
            using (NamedPipeServerStream pipeStream = new NamedPipeServerStream(pipe_name))
            {
                Console.WriteLine("[Server] Pipe created {0}", pipeStream.GetHashCode());

                // Wait for a connection
                pipeStream.WaitForConnection();
                Console.WriteLine("[Server] Pipe connection established");

                using (StreamReader sr = new StreamReader(pipeStream))
                {
                    string temp;
                    // We read a line from the pipe and print it together with the current time
                    while ((temp = sr.ReadLine()) != null || true)
                    {
                        if (temp != null)
                        {
                            Console.WriteLine("{0}: {1}", DateTime.Now, temp);
                        }
                    }
                }
            }

            Console.WriteLine("Connection lost");
        }

    }
}
