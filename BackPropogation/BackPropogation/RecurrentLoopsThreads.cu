#include "ReccurentLoops.cuh"

using namespace boost::interprocess;


inline boost::interprocess::managed_shared_memory open_shared_memory_object(string shared_memory_name){
	managed_shared_memory shared_memory(open_only, shared_memory_name.c_str());
	return shared_memory;
}

bool ReccurentLoops::start_training_threads(){
	this->initialize_threads();

	return true;
}


bool ReccurentLoops::stop_training_thread(){
	//Shut down any threads
	
	try{
		if (this->thread_list.size() > TIMER_THREAD){
			std::pair<bool*, std::size_t> bool_values;
			managed_shared_memory shared_timer(open_only, TIMER_SHARED);
			bool_values = shared_timer.find<bool>(TIMER_NEEDED);
			std::memset(bool_values.first, (bool)1, bool_values.second);
			this->thread_list[TIMER_THREAD]->join();//Wait for the thread to finish closing

			bool_values = shared_timer.find<bool>(PIPE_NEEDED);
			this->thread_list[PIPE_THREAD]->join();
		
			this->thread_list.clear();
		}
	}
	catch(exception e){

	}
	
	try{//Remove any shared_memory locations
		shared_memory_object::remove(TIMER_SHARED);
	}
	catch (exception e){

	}

	return true;
}


void timer_thread(NetworkTimer old_timer,string shared){
	try{
		managed_shared_memory shared_timer(open_only, shared.c_str());
	
	NetworkTimer timer = NetworkTimer(old_timer);
	std::pair<bool*, std::size_t> bool_values;
	timer.start();
	bool* temp = shared_timer.find<bool>(TIMER_NEEDED).first;

	while (*(shared_timer.find<bool>(TIMER_NEEDED).first) == true){
		if (timer.passed_checkpoint()){//A checkpoint should be created
			bool_values = shared_timer.find<bool>(CHECKPOINT_TIMER);
			std::memset(bool_values.first,(bool)1,bool_values.second);//Set the memory to true
			cout << true;
		}

		if (*(shared_timer.find<bool>(TIMER_PRINT).first)==true){
			timer.restart_timer();
			cout << timer.estimated_time_remaining(*(shared_timer.find<std::istream::streampos>(TIMER_PRINT_VALUE).first)) << endl;
			timer.clear_timer();
			bool_values = shared_timer.find<bool>(TIMER_PRINT);
			std::memset(bool_values.first, (bool)0, bool_values.second);//Set the memory to false, as it has been printed
		}

	}
	}
	catch (interprocess_exception e){
		cout << e.what();
		exit(0);
	}

}


void pipe_thread(string pipe_name,string shared){
	
	try{
		Connection_Pipe pipe = Connection_Pipe(pipe_name);
		managed_shared_memory shared_mem(open_only, shared.c_str());
		pipe.Open();
		char * message;
		char * message_list;
		std::pair<int*, size_t> int_pair;
		bool temp = false;
		while (*(shared_mem.find<bool>(PIPE_NEEDED).first) == true){

			if (pipe.has_new_message()){//Read a message
				message = const_cast<char*>(pipe.read().c_str());
				message_list = strtok(message, "--");
				if (strcmp(message_list, "Menu")==0)
				{
					int_pair = shared_mem.find<int>(SPECIAL_FUNCTIONS);
					std::memset(int_pair.first,std::stoi(strtok(NULL,"--")), int_pair.second);//Write the new value to memory
				}
				
			}
			try{
				if (!temp){
					cout << pipe.write("hello world\n");
					temp = true;
				}
				
			}
			catch (exception e){
				cout << e.what();
			}

		}
		pipe.Close();
	}
	catch (exception e){
		cout << e.what();
	}
	
}


void ReccurentLoops::initialize_threads(){
	this->thread_list = std::vector<thread*>(FINAL_THREAD_POS);
	
	managed_shared_memory managed{ open_only, TIMER_SHARED};
	
	managed.construct<bool>(TIMER_NEEDED)(true);
	managed.construct<bool>(TIMER_PRINT)(false);
	managed.construct<bool>(CHECKPOINT_TIMER)(false);
	managed.construct<std::istream::streampos>(TIMER_PRINT_VALUE)(std::istream::streampos());
	managed.construct<int>(SPECIAL_FUNCTIONS)(-1);
	managed.construct<bool>(PIPE_NEEDED)(true);
	
	this->thread_list[TIMER_THREAD] = new thread(timer_thread, this->timer, TIMER_SHARED);
	this->thread_list[PIPE_THREAD] = new thread(pipe_thread,"temp_pipe",TIMER_SHARED);
	
}

