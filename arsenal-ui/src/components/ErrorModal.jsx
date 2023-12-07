import React from 'react';
import { Modal, ModalBody, ModalHeader, ModalFooter, Button } from 'reactstrap';

export default class ErrorModal extends React.Component {

    render () {
        return (
            <div>
                <Modal isOpen={this.props.show} toggle={this.props.onHide}>
                <ModalHeader>{this.props.title}</ModalHeader>
                {this.props.error && 
                <ModalBody>
                    {this.props.error.message}
                </ModalBody>}
                <ModalFooter>
                    <Button color="secondary" onClick={this.props.onHide}>OK</Button>
                </ModalFooter>
                </Modal>
            </div>
        );
    }

}