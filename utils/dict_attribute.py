
class DictAsAttributes:
    def __init__(self, data_dict):
        self.__dict__['_data_dict'] = data_dict

    def __getattr__(self, key):
        if key in self._data_dict:
            return self._data_dict[key]
        else:
            raise AttributeError(f"'DictAsAttributes' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self._data_dict[key] = value

    def __delattr__(self, key):
        del self._data_dict[key]