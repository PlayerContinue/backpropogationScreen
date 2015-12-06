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
			this->thread_list.erase(this->thread_list.begin() + TIMER_THREAD);
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

		if (static_cast<bool>(shared_timer.find<bool>("PRINT_TIMER").first)){
			timer.restart_timer();
			//cout << timer.estimated_time_remaining(reinterpret_cast<std::istream::streampos>(file_remaining.get_address()));
			timer.clear_timer();
		}

	}
	}
	catch (interprocess_exception e){
		cout << e.what();
		exit(0);
	}

}


void pipe_thread(){

}


void ReccurentLoops::initialize_threads(){
	this->thread_list = std::vector<thread*>(FINAL_THREAD_POS);
	
	managed_shared_memory managed{ open_only, TIMER_SHARED};
	this->timer_shared_memory = &managed;
	
	timer_shared_memory->construct<bool>(TIMER_NEEDED)(true);
	timer_shared_memory->construct<bool>("PRINT_TIMER")(false);
	timer_shared_memory->construct<bool>(CHECKPOINT_TIMER)(false);
	this->thread_list[TIMER_THREAD] = new thread(timer_thread, this->timer, TIMER_SHARED);
	
}

