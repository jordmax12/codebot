const userController = require('./controllers/userController');

const getUser = async (event) => {
  try {
    const id = event.pathParameters.id;
    const user = await userController.getUserById(id);
    return {
      statusCode: 200,
      body: JSON.stringify(user),
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};

module.exports = {
  getUser,
};
