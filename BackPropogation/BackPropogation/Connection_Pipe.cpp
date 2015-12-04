#include "Connection_Pipe.h"


Connection_Pipe::Connection_Pipe()
{

	this->hPipeIn = create_pipe(L"Hyper_Pipe_IN");
	this->hPipeOut = create_pipe(L"Hyper_Pipe_OUT");
}

Connection_Pipe::Connection_Pipe(string pipe_name)
{

	//Create the pipe
	this->hPipeIn = create_pipe(convert_to_wstring(pipe_name + "_IN"));
	this->hPipeOut = create_pipe(convert_to_wstring(pipe_name + "_OUT"));
}

void Connection_Pipe::Open(){
	ConnectNamedPipe(this->hPipeIn, NULL);
	ConnectNamedPipe(this->hPipeOut, NULL);
}

void Connection_Pipe::Close(){
	CloseHandle(this->hPipeIn);
}

bool Connection_Pipe::add_to_write_queue(string insert){
	this->SendQueue.push(this->convert_to_wstring(insert).c_str());
	return true;
}

string Connection_Pipe::read_from_queue(){
	string front = this->RecieveQueue.front();
	this->RecieveQueue.pop();
	return front;
}

int Connection_Pipe::write(wstring toWrite){
	DWORD bytesWritten = 0;
	const wchar_t* data = toWrite.c_str();
	WriteFile(this->hPipeOut, data, _tcslen(data) * sizeof(const wchar_t*), &bytesWritten, NULL);
	return (int)bytesWritten;
}


HANDLE Connection_Pipe::create_pipe(wstring pipe_name){
	pipe_name = TEXT("\\\\.\\pipe\\") + pipe_name;//Combine the name with the general name of a pipe
	return ::CreateNamedPipe(pipe_name.c_str(),
		PIPE_ACCESS_DUPLEX,
		PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE,
		PIPE_UNLIMITED_INSTANCES,
		4096,
		4096,
		0,
		NULL);
}

wstring Connection_Pipe::convert_to_wstring(string convert){
	//Convert the string to wstring 
	std::wstring ws;
	ws.assign(convert.begin(), convert.end());
	return ws;
}

string Connection_Pipe::read(){
	return "";
}